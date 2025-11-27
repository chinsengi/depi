#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modified by Shirui Chen, 2025
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
import importlib
from typing import Any

import gymnasium as gym

from lerobot.common.envs.configs import (
    AlohaEnv,
    EnvConfig,
    LiberoEnv,
    MetaworldEnv,
    PushtEnv,
    XarmEnv,
)
from lerobot.common.processor import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    LiberoEnvProcessorStep,
)
from lerobot.common.processor.converters import observation_to_transition, transition_to_observation


class UnsupportedRemoteEnvError(RuntimeError):
    """Raised when attempting to build a remote environment in this distribution."""

    pass


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    elif env_type == "metaworld":
        return MetaworldEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig | str): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.
        hub_cache_dir (str | None): kept for parity with upstream signature; not supported here.
        trust_remote_code (bool): kept for parity with upstream signature; not supported here.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed
        UnsupportedRemoteEnvError: If a Hugging Face Hub environment string is provided.

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]: Mapping of suite name to vectorized envs.
    """
    if isinstance(cfg, str):
        raise UnsupportedRemoteEnvError("Hub-provided environments are not supported in this distribution.")

    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    if "libero" in cfg.type:
        from lerobot.common.envs.libero import create_libero_envs

        if cfg.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            camera_name_mapping=cfg.camera_name_mapping,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    if "metaworld" in cfg.type:
        from lerobot.common.envs.metaworld import create_metaworld_envs

        if cfg.task is None:
            raise ValueError("MetaWorld requires a task to be specified")

        return create_metaworld_envs(
            task=cfg.task,
            n_envs=n_envs,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    package_name = f"gym_{cfg.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    def _make_one() -> gym.Env:
        return gym.make(gym_handle, disable_env_checker=True, **cfg.gym_kwargs)

    vec_env = env_cls(
        [_make_one for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    return {cfg.type: {0: vec_env}}


def make_env_pre_post_processors(
    env_cfg: EnvConfig,
) -> tuple[
    DataProcessorPipeline[dict[str, Any], dict[str, Any]],
    DataProcessorPipeline[dict[str, Any], dict[str, Any]],
]:
    """Create environment-specific preprocessor and postprocessor pipelines.

    Args:
        env_cfg: The environment configuration

    Returns:
        A tuple of (env_preprocessor, env_postprocessor) pipelines
    """
    # Determine processor steps based on environment type
    preprocessor_steps = []

    # LIBERO and Metaworld need agent_pos handling for pixels-only mode
    if env_cfg.type in ("libero", "metaworld"):
        # Different agent_pos dimensions for each environment
        agent_pos_dim = 8 if env_cfg.type == "libero" else 4
        preprocessor_steps.append(LiberoEnvProcessorStep(agent_pos_dim=agent_pos_dim))
    else:
        # Other environments use identity processor
        preprocessor_steps.append(IdentityProcessorStep())

    env_preprocessor = DataProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=preprocessor_steps,
        name="env_preprocessor",
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Postprocessor is identity for all environments
    env_postprocessor = DataProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[IdentityProcessorStep()],
        name="env_postprocessor",
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    return env_preprocessor, env_postprocessor
