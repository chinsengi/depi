#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Helpers for manipulating batched transitions."""

from __future__ import annotations

from typing import TypedDict

import torch

from lerobot.common.utils.constants import ACTION


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float | torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: bool | torch.Tensor
    truncated: bool | torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None


def move_transition_to_device(transition: Transition, device: str = "cpu") -> Transition:
    """Recursively move a transition to the target device."""
    tgt = torch.device(device)
    non_blocking = tgt.type == "cuda"
    transition["state"] = {key: tensor.to(tgt, non_blocking=non_blocking) for key, tensor in transition["state"].items()}
    transition[ACTION] = transition[ACTION].to(tgt, non_blocking=non_blocking)
    if isinstance(transition["reward"], torch.Tensor):
        transition["reward"] = transition["reward"].to(tgt, non_blocking=non_blocking)
    if isinstance(transition["done"], torch.Tensor):
        transition["done"] = transition["done"].to(tgt, non_blocking=non_blocking)
    if isinstance(transition["truncated"], torch.Tensor):
        transition["truncated"] = transition["truncated"].to(tgt, non_blocking=non_blocking)
    transition["next_state"] = {
        key: tensor.to(tgt, non_blocking=non_blocking) for key, tensor in transition["next_state"].items()
    }
    comp_info = transition.get("complementary_info")
    if comp_info is not None:
        for key, value in comp_info.items():
            if isinstance(value, torch.Tensor):
                comp_info[key] = value.to(tgt, non_blocking=non_blocking)
            elif isinstance(value, (int, float, bool)):
                comp_info[key] = torch.tensor(value, device=tgt)
            else:
                raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")
    return transition


def move_state_dict_to_device(data, device: str = "cpu"):
    """Move every tensor within nested structures to the target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_state_dict_to_device(v, device=device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_state_dict_to_device(v, device=device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in data)
    return data
