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

"""Utility helpers for control loops and live data collection."""

from __future__ import annotations

import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot


@cache
def is_headless() -> bool:
    """Return True if the environment lacks graphical capabilities."""
    try:  # noqa: PLC0415 - import guarded by cache
        import pynput  # type: ignore  # noqa: F401

        return False
    except Exception:  # pragma: no cover - depends on environment availability
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> PolicyAction:
    """Run a single forward pass of the policy and return the processed action."""
    observation = copy(observation)
    autocast_ctx = (
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        prepared_obs = prepare_observation_for_inference(observation, device, task, robot_type)
        processed_obs = preprocessor(prepared_obs)
        action = policy.select_action(processed_obs)
        action = postprocessor(action)
    return action


def init_keyboard_listener():
    """Register non-blocking keyboard shortcuts for interactive control."""
    events: dict[str, bool] = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        return None, events

    from pynput import keyboard  # type: ignore # noqa: PLC0415

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error handling key press: {exc}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def sanity_check_dataset_name(repo_id: str, policy_cfg) -> None:
    """Validate dataset naming convention against the presence of a policy config."""
    _, dataset_name = repo_id.split("/", maxsplit=1)
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg})."
        )
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """Ensure recorded metadata matches the dataset being appended to."""
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches: list[str] = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
