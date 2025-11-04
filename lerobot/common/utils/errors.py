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

"""Common exception classes for device lifecycle management."""

from __future__ import annotations


class DeviceNotConnectedError(ConnectionError):
    """Raised when attempting to use a device that is not connected."""

    def __init__(self, message: str = "This device is not connected. Try calling `connect()` first.") -> None:
        super().__init__(message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Raised when attempting to connect an already connected device."""

    def __init__(
        self,
        message: str = "This device is already connected. Try not calling `connect()` twice.",
    ) -> None:
        super().__init__(message)
