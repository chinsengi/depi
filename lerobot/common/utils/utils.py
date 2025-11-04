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
import gc
import logging
import os
import os.path as osp
import platform
import select
import subprocess
import sys
import time
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

try:  # optional dependency
    from accelerate import Accelerator  # type: ignore
except ImportError:  # pragma: no cover - accelerate not installed
    Accelerator = None  # type: ignore

from datasets.utils.logging import disable_progress_bar, enable_progress_bar


def none_or_int(value: str | int | None) -> int | None:
    if value == "None":
        return None
    return int(value) if value is not None else None


def inside_slurm() -> bool:
    """Check whether the python process was launched through slurm."""
    # TODO(rcadene): return False for interactive mode `--pty bash`
    return "SLURM_JOB_ID" in os.environ


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    if _has_xpu():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
    return torch.device("cpu")


def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on availability."""
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA device requested but not available"
        device = torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available(), "MPS device requested but not available"
        device = torch.device("mps")
    elif try_device == "xpu":
        assert _has_xpu(), "XPU device requested but not available"
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning("Using custom %s device.", try_device)
    return device


def get_best_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if _has_xpu():
        return torch.device("xpu")
    logging.warning("Using CPU since no accelerated device is available; training will be slow.")
    return torch.device("cpu")


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """Ensure dtype is compatible with the given device."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if _has_xpu() and hasattr(torch.xpu, "get_device_capability"):
            capability = torch.xpu.get_device_capability()
            if not capability.get("has_fp64", False):
                logging.warning("Device %s does not support float64, using float32 instead.", device)
                return torch.float32
        else:
            logging.warning("Unable to query XPU FP64 capability; defaulting to float32.")
            return torch.float32
    return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    if try_device == "mps":
        return torch.backends.mps.is_available()
    if try_device == "xpu":
        return _has_xpu()
    if try_device == "cpu":
        return True
    raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str) -> bool:
    if device in ["cuda", "cpu", "xpu"]:
        return True
    if device == "mps":
        return False
    raise ValueError(f"Unknown device '{device}.")


def init_logging(
    accelerator: Any = None,
    only_main: bool = False,
    *,
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Initialize logging configuration for LeRobot.

    Args:
        accelerator: Optional accelerator instance used to determine main process.
        only_main: If True, suppress console logging on non-main processes.
        log_file: Optional file path to write logs to.
        display_pid: Include process ID in log messages.
        console_level: Logging level for console output.
        file_level: Logging level for file output.
    """

    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)

    is_main_process = True
    if accelerator is not None and hasattr(accelerator, "is_main_process"):
        is_main_process = accelerator.is_main_process

    formatter = logging.Formatter()
    formatter.format = custom_format

    if not (only_main and not is_main_process):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)


def format_big_number(num: float, precision: int = 0) -> str:
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0
    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor
    return f"{num:.{precision}f}{suffixes[-1]}"


def say(text: str, blocking: bool = False) -> None:
    system = platform.system()
    if system == "Darwin":
        cmd = ["say", text]
    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")
    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]
    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        creationflags = subprocess.CREATE_NO_WINDOW if system == "Windows" else 0
        subprocess.Popen(cmd, creationflags=creationflags)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
    logging.info(text)
    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)
    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    try:
        np.dtype(dtype_str)
        return True
    except TypeError:
        return False


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")
        return False
    return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines: int) -> None:
    print(f"\033[{lines}A", end="")


def get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time_s: float) -> tuple[int, int, int, float]:
    days = int(elapsed_time_s // (24 * 3600))
    elapsed_time_s %= 24 * 3600
    hours = int(elapsed_time_s // 3600)
    elapsed_time_s %= 3600
    minutes = int(elapsed_time_s // 60)
    seconds = elapsed_time_s % 60
    return days, hours, minutes, seconds


def _relative_path_between(path1: Path, path2: Path) -> Path:
    path1 = path1.absolute()
    path2 = path2.absolute()
    try:
        return path1.relative_to(path2)
    except ValueError:
        common_parts = Path(osp.commonpath([path1, path2])).parts
        return Path(
            "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
        )


def print_cuda_memory_usage() -> None:
    """Use this function to locate and debug memory leaks."""
    if not torch.cuda.is_available():  # pragma: no cover - guard when cuda missing
        logging.info("CUDA is not available; cannot report memory usage.")
        return
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Current GPU Memory Allocated: %.2f MB", torch.cuda.memory_allocated(0) / 1024**2)
    logging.info("Maximum GPU Memory Allocated: %.2f MB", torch.cuda.max_memory_allocated(0) / 1024**2)
    logging.info("Current GPU Memory Reserved: %.2f MB", torch.cuda.memory_reserved(0) / 1024**2)
    logging.info("Maximum GPU Memory Reserved: %.2f MB", torch.cuda.max_memory_reserved(0) / 1024**2)


def capture_timestamp_utc() -> datetime:
    return datetime.now(timezone.utc)


class SuppressProgressBars:
    """Context manager to temporarily disable datasets progress bars."""

    def __enter__(self) -> None:
        disable_progress_bar()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        enable_progress_bar()


class TimerManager:
    """Lightweight utility to measure elapsed time."""

    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self) -> "TimerManager":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> "TimerManager":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            message = f"{self.label}: {elapsed:.6f} s"
            if self.logger is not None:
                self.logger.info(message)
            else:
                logging.info(message)
        return elapsed

    def reset(self) -> None:
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:
        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:
        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val
