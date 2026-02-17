#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from huggingface_hub.constants import CONFIG_NAME

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

if TYPE_CHECKING:
    from lerobot.common.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: Any | None = None  # RTCConfig if available

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings (excluded from saved config for inference compatibility)
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)
    attention_implementation: str = "eager"  # Attention implementation: eager, sdpa, flash_attention_2

    # LoRA settings (excluded from saved config for inference compatibility)
    use_lora: bool = False  # Enable LoRA (Low-Rank Adaptation) for efficient fine-tuning
    lora_rank: int = 16  # Rank of LoRA matrices (r in the paper)
    lora_alpha: float = 32.0  # LoRA scaling factor (usually 2x rank)
    lora_dropout: float = 0.1  # Dropout probability for LoRA layers
    lora_target_modules: list[str] | None = None  # Target modules for LoRA (None = auto-detect)

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Fields that are training-only and should not be saved in the model config
    # (they cause errors when loading in inference-only codebases)
    _TRAINING_ONLY_FIELDS: ClassVar[tuple] = (
        "compiled", "attention_implementation", "compile_model", "compile_mode",
        "gradient_checkpointing",
        "use_lora", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules",
        "optimizer_lr", "optimizer_betas", "optimizer_eps", "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_warmup_steps", "scheduler_decay_steps", "scheduler_decay_lr",
    )

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config excluding training-only fields for inference compatibility."""
        import draccus

        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

        # Re-read and strip training-only fields
        config_path = save_directory / CONFIG_NAME
        with open(config_path) as f:
            config_dict = json.load(f)
        for field_name in self._TRAINING_ONLY_FIELDS:
            config_dict.pop(field_name, None)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
