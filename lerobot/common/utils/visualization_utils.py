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

"""Rerun-based visualization helpers for live control sessions."""

from __future__ import annotations

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from lerobot.common.utils.constants import OBS_PREFIX, OBS_STR


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initialize the Rerun SDK with sane defaults."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """Log observation and action dictionaries to the Rerun viewer."""
    if observation:
        for key, value in observation.items():
            if value is None:
                continue
            rerun_key = key if str(key).startswith(OBS_PREFIX) else f"{OBS_STR}.{key}"
            _log_value(rerun_key, value)

    if action:
        for key, value in action.items():
            if value is None:
                continue
            rerun_key = key if str(key).startswith("action.") else f"action.{key}"
            _log_value(rerun_key, value)


def _log_value(key: str, value: Any) -> None:
    if _is_scalar(value):
        rr.log(key, rr.Scalars(float(value)))
        return

    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 1:
            for idx, element in enumerate(array):
                rr.log(f"{key}_{idx}", rr.Scalars(float(element)))
        else:
            rr.log(key, rr.Image(array), static=True)
        return

    # Fallback: flatten iterable structures for logging
    if isinstance(value, (list, tuple)):
        for idx, element in enumerate(value):
            _log_value(f"{key}_{idx}", element)
        return

    raise TypeError(f"Unsupported type {type(value)} for Rerun logging")
