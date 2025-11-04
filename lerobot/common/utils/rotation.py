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

"""Custom rotation utilities that mirror a subset of scipy.spatial.transform.Rotation."""

from __future__ import annotations

import numpy as np


class Rotation:
    """Simple representation of 3D rotations backed by quaternions."""

    def __init__(self, quat: np.ndarray) -> None:
        self._quat = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(self._quat)
        if norm > 0:
            self._quat = self._quat / norm

    # Constructors -----------------------------------------------------------------

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "Rotation":
        rotvec = np.asarray(rotvec, dtype=float)
        angle = np.linalg.norm(rotvec)
        if angle < 1e-8:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = rotvec / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            quat = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])
        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Rotation":
        matrix = np.asarray(matrix, dtype=float)
        trace = np.trace(matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw])
        return cls(quat)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "Rotation":
        return cls(quat)

    # Representation ----------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        qx, qy, qz, qw = self._quat
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    def as_rotvec(self) -> np.ndarray:
        qx, qy, qz, qw = self._quat
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        angle = 2.0 * np.arccos(np.clip(abs(qw), 0.0, 1.0))
        sin_half_angle = np.sqrt(1.0 - qw * qw)
        if sin_half_angle < 1e-8:
            return 2.0 * np.array([qx, qy, qz])
        axis = np.array([qx, qy, qz]) / sin_half_angle
        return angle * axis

    def as_quat(self) -> np.ndarray:
        return self._quat.copy()

    # Transformations ----------------------------------------------------------------

    def apply(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=float)
        original_shape = vectors.shape
        single_vector = False
        if vectors.ndim == 1:
            if vectors.shape[0] != 3:
                raise ValueError("Single vector must have length 3")
            vectors = vectors.reshape(1, 3)
            single_vector = True
        elif vectors.shape[-1] != 3:
            raise ValueError("Input vectors must have shape (..., 3)")

        matrix = self.as_matrix()
        if inverse:
            matrix = matrix.T
        rotated = vectors @ matrix.T
        if single_vector:
            rotated = rotated.reshape(original_shape)
        return rotated
