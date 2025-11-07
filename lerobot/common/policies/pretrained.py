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
import abc
import logging
import os
from pathlib import Path
from typing import List, Tuple, Type, TypeVar, Union

import packaging
import safetensors
import torch
from accelerate.utils import is_compiled_module
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import _remove_duplicate_names
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.common.utils.hub import HubMixin
from lerobot.configs.policies import PreTrainedConfig

T = TypeVar("T", bound="PreTrainedPolicy")

DEFAULT_POLICY_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This policy has been pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot):
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


def load_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
    device: Union[str, int] = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Loads a given state dictionary onto a torch model.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to load onto.
        state_dict (`dict[str, torch.Tensor]`):
            The state dictionary to load onto the model.
        strict (`bool`, *optional*, defaults to True):
            Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `(missing, unexpected): (List[str], List[str])`
            `missing` are names in the model which were not modified during loading
            `unexpected` are names that are on the file, but weren't used during
            the load.
    """
    # state_dict = load_file(filename, device=device)
    model_state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(model_state_dict, preferred_names=state_dict.keys())

    reverse_to_remove = {}
    for key, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            reverse_to_remove[to_remove] = key

    # We iterate on the model, so we'll add keys we find missing
    # here
    missing = set()
    # We start with all keys on disk declared as unexpected, we'll
    # slowly remove them when we find them
    unexpected = set(state_dict.keys())
    # Some keys can be invalid too.
    invalid = set()

    for k, mv in model_state_dict.items():
        actual_k = reverse_to_remove.get(k)
        look_k = actual_k if actual_k is not None else k
        v = state_dict.get(look_k)
        if v is None:
            missing.add(k)
        else:
            # We can actually check for the shapes while we're at it.
            # For the device, it's trickier given torch's internals
            # There might be some Meta device for faster initiation
            if v.dtype != mv.dtype or v.shape != mv.shape:
                invalid.add(k)
            if actual_k is None:
                unexpected.remove(k)

    missing = set(missing)
    unexpected = set(unexpected)
    if strict and (missing or unexpected or invalid):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        invalid_keys = ", ".join([f'"{k}"' for k in sorted(invalid)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        if invalid:
            error += f"\n    Invalid key(s) in state_dict: {invalid_keys}, mismatched dtypes or shape."
        del state_dict
        raise RuntimeError(error)

    torch_missing, torch_unexpected = model.load_state_dict(state_dict, strict=False)
    # Sanity check that the work we've done matches
    # Pytorch internal loading.
    torch_missing = set(torch_missing)
    torch_unexpected = set(torch_unexpected)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in torch_missing:
                torch_unexpected.add(to_remove)
            else:
                torch_missing.remove(to_remove)
    assert torch_missing == missing, f"{torch_missing} != {missing}"
    assert torch_unexpected == unexpected, f"{torch_unexpected} != {unexpected}"
    return missing, unexpected


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        compile = kwargs.pop("compile", False)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            # TODO: when loading from a compiled model checkpoint, we need to compile the model first to correctly load the weights
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            if not os.path.exists(model_file):
                logging.warning(f"Model file {model_file} not found, using the original initialization.")
                policy = instance
            else:
                state_dict_to_load = safetensors.torch.load_file(model_file)
                uncompiled_state_dict = {}
                for key, value in state_dict_to_load.items():
                    uncompiled_key = key.replace("._orig_mod", "")
                    uncompiled_key = uncompiled_key.replace(".module", "")
                    uncompiled_state_dict[uncompiled_key] = value
                model_file = Path(model_file)
                safetensors.torch.save_file(
                    uncompiled_state_dict, model_file.parent / "uncompiled_model.safetensors"
                )
                uncompiled_model_file = model_file.parent / "uncompiled_model.safetensors"
                policy = cls._load_as_safetensor(instance, uncompiled_model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e
            try:
                # try normal loading first
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except Exception:
                logging.warning(f"Failed to load model from {model_file}, trying to compile the model.")
                instance.model = torch.compile(instance.model)
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)

        if compile and not is_compiled_module(instance.model):
            policy.model = torch.compile(policy.model)

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logging.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            state_dict = safetensors.torch.load_file(model_file, device=map_location)
            normalizing_weights = {key: state_dict[key] for key in state_dict if "normalize" in key}
            try:
                model.load_state_dict(normalizing_weights, strict=False)
            except Exception as e:
                print(f"Error loading normalizing weights: {e}")
                # delete the normalizing weights if the new input dimension is not multiple of the checkpoint
                keys = list(normalizing_weights.keys())
                if any(
                    state_dict[key].shape[0] % model.config.output_features["action"].shape[0] != 0
                    for key in keys
                ):
                    strict = False
                    for key in normalizing_weights:
                        del state_dict[key]
                else:
                    # Or we can tile the normalizing weights to fit bimanual
                    for key in normalizing_weights:
                        state_dict[key] = normalizing_weights[key].repeat(2)
            load_model(model, state_dict, strict=strict, device=map_location)
            # safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)

        return model

    # def generate_model_card(self, *args, **kwargs) -> ModelCard:
    #     card = ModelCard.from_template(
    #         card_data=self._hub_mixin_info.model_card_data,
    #         template_str=self._hub_mixin_info.model_card_template,
    #         repo_url=self._hub_mixin_info.repo_url,
    #         docs_url=self._hub_mixin_info.docs_url,
    #         **kwargs,
    #     )
    #     return card

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
