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
import shutil
from pathlib import Path

import torch
from accelerate.utils import is_compiled_module
from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from lerobot.common.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.common.datasets.utils import load_json, write_json
from lerobot.common.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.common.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import load_rng_state, save_rng_state
from lerobot.configs.train import TrainPipelineConfig

# Additional constants for dataloader state
DATALOADER_STATE = "dataloader_state.pth"


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def cleanup_old_checkpoints(checkpoints_dir: Path, keep_last_n: int = 2) -> None:
    """
    Clean up old checkpoints, keeping only the last N checkpoints.

    Args:
        checkpoints_dir (Path): Directory containing checkpoints
        keep_last_n (int): Number of most recent checkpoints to keep
    """
    if not checkpoints_dir.exists():
        return

    # Get all checkpoint directories (excluding the 'last' symlink)
    checkpoint_dirs = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and item.name != LAST_CHECKPOINT_LINK:
            try:
                # Try to parse the step number from the directory name
                step_num = int(item.name)
                checkpoint_dirs.append((step_num, item))
            except ValueError:
                # Skip directories that don't have numeric names
                continue

    if len(checkpoint_dirs) <= keep_last_n:
        return

    # Sort by step number and keep only the most recent ones
    checkpoint_dirs.sort(key=lambda x: x[0])
    checkpoints_to_remove = checkpoint_dirs[:-keep_last_n]

    for step_num, checkpoint_dir in checkpoints_to_remove:
        logging.info(f"Removing old checkpoint: {checkpoint_dir.name} (step {step_num})")
        shutil.rmtree(checkpoint_dir)


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    dataloader=None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   └── train_config.json  # train config
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        ├── dataloader_state.json  # dataloader state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        dataloader: The DataLoader instance to save state from. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler, dataloader)


def save_dataloader_state(dataloader: DataLoader, save_dir: Path) -> None:
    """
    Saves the dataloader state including sampler state and iteration position.

    Args:
        dataloader: The DataLoader instance to save state from.
        save_dir (Path): The directory to save the dataloader state to.
    """
    torch.save(dataloader.state_dict(), save_dir / DATALOADER_STATE)


def load_dataloader_state(dataloader, load_dir: Path) -> None:
    """
    Loads the dataloader state including sampler state and iteration position.

    Args:
        dataloader: The DataLoader instance to load state into.
        load_dir (Path): The directory to load the dataloader state from.
    """
    dataloader.load_state_dict(torch.load(load_dir / DATALOADER_STATE, weights_only=True))


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    dataloader=None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, rng state, and dataloader state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
        dataloader: The DataLoader instance to save state from. Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)
    if dataloader is not None:
        save_dataloader_state(dataloader, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None, dataloader=None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, rng state, and dataloader state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).
        dataloader: The DataLoader instance to load state into. Defaults to None.

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)
    if dataloader is not None:
        load_dataloader_state(dataloader, training_state_dir)

    return step, optimizer, scheduler


def reconcile_checkpoint_compile_state(checkpoint_path: Path, model: torch.nn.Module) -> None:
    """
    Align checkpoint weights with current model regarding torch.compile wrapping.
    If the checkpoint was saved from an uncompiled model, keys won't have the `_orig_mod.` prefix.
    If it was saved from a compiled model, keys will start with `_orig_mod.`.
    Adjust the checkpoint in-place when the checkpoint and current model differ.
    """
    candidate_files = [
        ("safetensors", checkpoint_path / "model.safetensors"),
        ("safetensors", checkpoint_path / "pytorch_model.safetensors"),
        ("bin", checkpoint_path / "model.bin"),
        ("bin", checkpoint_path / "pytorch_model.bin"),
    ]
    model_file = None
    save_format = None
    for fmt, path in candidate_files:
        if path.exists():
            model_file, save_format = path, fmt
            break
    if model_file is None:
        return

    if save_format == "safetensors":
        from safetensors.torch import load_file, save_file

        state_dict = load_file(str(model_file))
    else:
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)

    def _has_orig_mod(keys):
        return any("._orig_mod." in k or k.startswith("_orig_mod.") for k in keys)

    state_compiled = _has_orig_mod(state_dict.keys())
    model_compiled = is_compiled_module(model)

    if state_compiled == model_compiled:
        return

    def _strip_orig_mod(key: str) -> str:
        if "._orig_mod." in key:
            return key.replace("._orig_mod.", ".", 1)
        if key.startswith("_orig_mod."):
            return key.removeprefix("_orig_mod.")
        return key

    def _add_orig_mod(key: str) -> str:
        if "._orig_mod." in key or key.startswith("_orig_mod."):
            return key
        # common Hugging Face file naming uses a leading "model." prefix
        if key.startswith("model."):
            return key.replace("model.", "model._orig_mod.", 1)
        return f"_orig_mod.{key}"

    if state_compiled:
        patched_state = {_strip_orig_mod(k): v for k, v in state_dict.items()}
        logging.info("Checkpoint de-prefixed for uncompiled model.")
    else:
        patched_state = {_add_orig_mod(k): v for k, v in state_dict.items()}
        logging.info("Checkpoint prefixed for compiled model.")

    if save_format == "safetensors":
        from safetensors.torch import save_file  # type: ignore

        save_file(patched_state, str(model_file))
    else:
        torch.save(patched_state, model_file)
