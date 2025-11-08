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
import json
import logging
from pprint import pformat

import packaging.version
import torch
from tqdm import tqdm

from data_ids.filter_so100_data import get_repo_ids
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.lerobot_dataset_v3 import (
    LeRobotDatasetMetadataV3,
    LeRobotDatasetV3,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.utils import get_repo_versions
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata | LeRobotDatasetMetadataV3
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata | LeRobotDatasetMetadataV3): The dataset metadata providing
            features and fps against which delta_timestamps are constructed.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _parse_revision_to_version(revision: str | None) -> packaging.version.Version | None:
    if revision is None:
        return None
    normalized = revision.lstrip("v")
    try:
        return packaging.version.parse(normalized)
    except packaging.version.InvalidVersion:
        return None


def _infer_dataset_major_version(repo_id: str, cfg: TrainPipelineConfig) -> int | None:
    version_from_revision = _parse_revision_to_version(cfg.dataset.revision)
    if version_from_revision is not None:
        return version_from_revision.major

    repo_versions = get_repo_versions(
        repo_id,
        root=cfg.dataset.root,
        force_cache_sync=cfg.dataset.force_cache_sync,
    )
    if repo_versions:
        return max(repo_versions).major

    return None


def load_delta_timestamps(
    repo_id: str, cfg: TrainPipelineConfig, major_version: int | None = None
) -> dict[str, list] | None:
    """Loads delta timestamps for a given dataset repository ID based on the provided configuration.

    Args:
        repo_id (str): The repository ID of the dataset.
        cfg (TrainPipelineConfig): The configuration that contains dataset and policy settings.
        major_version (int | None): Optional pre-computed major version for the repo.

    Returns:
        dict[str, list] | None: A dictionary of delta timestamps or None if not applicable.
    """
    if major_version is None:
        major_version = _infer_dataset_major_version(repo_id, cfg)
    is_v3_dataset = major_version is not None and major_version >= 3

    if is_v3_dataset:
        ds_meta = LeRobotDatasetMetadataV3(
            repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            force_cache_sync=cfg.dataset.force_cache_sync,
        )
    else:
        if major_version is None:
            logging.warning(
                "Could not determine dataset version for %s; defaulting to v2.1 metadata.",
                repo_id,
            )
        ds_meta = LeRobotDatasetMetadata(
            repo_id,
            root=cfg.dataset.root,
            revision=cfg.dataset.revision,
            force_cache_sync=cfg.dataset.force_cache_sync,
            use_annotated_tasks=cfg.dataset.use_annotated_tasks,
        )
    return resolve_delta_timestamps(cfg.policy, ds_meta)


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        LeRobotDataset | MultiLeRobotDataset: A dataset instance that can be either a single dataset or a collection of datasets.
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if cfg.dataset.repo_id is not None:
        major_version = _infer_dataset_major_version(cfg.dataset.repo_id, cfg)
        delta_timestamps = load_delta_timestamps(cfg.dataset.repo_id, cfg, major_version=major_version)
        is_v3_dataset = major_version is not None and major_version >= 3

        if is_v3_dataset:
            dataset = LeRobotDatasetV3(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                force_cache_sync=cfg.dataset.force_cache_sync,
            )
        else:
            if major_version is None:
                logging.warning(
                    "Could not determine dataset version for %s; defaulting to v2.1 dataset loader.",
                    cfg.dataset.repo_id,
                )
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                force_cache_sync=cfg.dataset.force_cache_sync,
                use_annotated_tasks=cfg.dataset.use_annotated_tasks,
            )
    else:
        # Handle multiple datasets
        # TODO: support more flexible dataset selection
        if cfg.dataset.repo_ids is None:
            logging.info(f"Loading {cfg.num_datasets} pretraining datasets.")
            repo_ids = get_repo_ids("so100", len_limit=cfg.num_datasets, load_from_cache=True)
        else:
            logging.info(f"Loading datasets from {cfg.dataset.repo_ids}")
            with open(cfg.dataset.repo_ids) as f:
                repo_ids = json.load(f)
        delta_timestamps_dict = {}
        filtered_repo_ids = []
        skipped_repo_ids = []
        for repo_id in tqdm(repo_ids, desc="Processing datasets metadata"):
            try:
                delta_timestamps = load_delta_timestamps(repo_id, cfg)
            except Exception as e:
                logging.warning(
                    "Skipping dataset %s because delta timestamps could not be loaded: %s",
                    repo_id,
                    e,
                )
                skipped_repo_ids.append(repo_id)
                continue

            if not delta_timestamps:
                logging.warning(
                    "Skipping dataset %s because no delta timestamps were returned.",
                    repo_id,
                )
                skipped_repo_ids.append(repo_id)
                continue

            delta_timestamps_dict[repo_id] = delta_timestamps
            filtered_repo_ids.append(repo_id)

        if not filtered_repo_ids:
            raise RuntimeError("No datasets left after filtering invalid delta timestamps.")

        if skipped_repo_ids:
            logging.warning(
                "Skipped %d dataset(s) with missing delta timestamps. First few: %s",
                len(skipped_repo_ids),
                skipped_repo_ids[:5],
            )

        dataset = MultiLeRobotDataset(
            filtered_repo_ids,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps_dict,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            force_cache_sync=cfg.dataset.force_cache_sync,
            use_annotated_tasks=cfg.dataset.use_annotated_tasks,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        if isinstance(dataset, (LeRobotDataset, LeRobotDatasetV3)):
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        elif isinstance(dataset, MultiLeRobotDataset):
            for ds in dataset._datasets:
                try:
                    for key in ds.meta.camera_keys:
                        for stats_type, stats in IMAGENET_STATS.items():
                            ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
                except Exception as e:
                    print(f"Error processing dataset {ds.repo_id}: {e}")
                    continue
    return dataset
