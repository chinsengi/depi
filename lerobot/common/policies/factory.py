#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import env_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    IdentityProcessorStep,
    PolicyAction,
    RenameObservationsProcessorStep,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    compile: bool = False,
    strict: bool = True,
    device: torch.device | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.
        compile (bool, optional): Whether to compile the policy model. Defaults to False.
        strict (bool, optional): Whether to strictly enforce state dict loading. Defaults to True.
        device (torch.device | None, optional): Device to load the policy on. Defaults to None.
        rename_map (dict[str, str] | None, optional): Mapping for renaming observation keys. This parameter
            is accepted for API compatibility but not used directly by make_policy. Use make_pre_post_processors
            to create processors that use the rename_map. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: The instantiated policy
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        kwargs["compile"] = compile
        kwargs["strict"] = strict
        policy = policy_cls.from_pretrained(**kwargs)
        policy.to(device if device is not None else cfg.device)

        assert isinstance(policy, nn.Module)

        assert compile or cfg.attention_implementation != "flex", (
            "make sure compile is True when using flex attention"
        )
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

        policy.to(device if device is not None else cfg.device)
        assert isinstance(policy, nn.Module)

        assert compile or cfg.attention_implementation != "flex", (
            "make sure compile is True when using flex attention"
        )
        if compile:
            # please do not use other mode than the default one, will not work
            policy.model = torch.compile(policy.model)

    return policy


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | Path | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
) -> tuple[
    DataProcessorPipeline[dict[str, Any], dict[str, Any]], DataProcessorPipeline[PolicyAction, PolicyAction]
]:
    """Create preprocessor and postprocessor pipelines for a policy.

    Args:
        policy_cfg: The policy configuration
        pretrained_path: Path to pretrained model (if any)
        preprocessor_overrides: Dictionary of overrides for the preprocessor, e.g., {"rename_map": {...}}

    Returns:
        A tuple of (preprocessor, postprocessor) pipelines
    """
    preprocessor_steps = []

    # Add rename step if rename_map is provided
    if preprocessor_overrides and "rename_map" in preprocessor_overrides:
        rename_map = preprocessor_overrides["rename_map"]
        if rename_map:
            preprocessor_steps.append(RenameObservationsProcessorStep(rename_map=rename_map))

    # Add device transfer step
    device_str = str(policy_cfg.device) if hasattr(policy_cfg, "device") else "cpu"
    preprocessor_steps.append(DeviceProcessorStep(device=device_str))

    # If no steps were added, use identity
    if not preprocessor_steps:
        preprocessor_steps = [IdentityProcessorStep()]

    # Create preprocessor pipeline
    preprocessor = DataProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=preprocessor_steps, name="preprocessor"
    )

    # Create postprocessor pipeline (identity for now)
    postprocessor = DataProcessorPipeline[PolicyAction, PolicyAction](
        steps=[IdentityProcessorStep()], name="postprocessor"
    )

    return preprocessor, postprocessor
