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
from collections.abc import Iterator, Sequence

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict[str, Sequence[int]] | None = None,
        *,
        dataset_from_indices: Sequence[int] | None = None,
        dataset_to_indices: Sequence[int] | None = None,
        episode_indices_to_use: Sequence[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' describing episode spans.
                When provided, overrides ``dataset_from_indices`` and ``dataset_to_indices``.
            dataset_from_indices: List of indices marking the start of each episode in the dataset.
            dataset_to_indices: List of indices marking the end of each episode in the dataset.
            episode_indices_to_use: Iterable of episode indices to include. If None, all episodes are used.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if episode_data_index is not None:
            dataset_from_indices = episode_data_index["from"]
            dataset_to_indices = episode_data_index["to"]

        if dataset_from_indices is None or dataset_to_indices is None:
            raise ValueError(
                "Either `episode_data_index` or both `dataset_from_indices` and `dataset_to_indices` must be provided."
            )

        if len(dataset_from_indices) != len(dataset_to_indices):
            raise ValueError("`dataset_from_indices` and `dataset_to_indices` must have the same length.")

        # Normalize containers to simple Python ints to ease downstream processing.
        normalized_from = [int(idx) for idx in dataset_from_indices]
        normalized_to = [int(idx) for idx in dataset_to_indices]

        if episode_indices_to_use is not None:
            episode_indices_to_use = set(int(ep_idx) for ep_idx in episode_indices_to_use)

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(normalized_from, normalized_to, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
