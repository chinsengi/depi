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

import torch
from tqdm import tqdm
from data_ids.filter_so100_data import get_repo_ids

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

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
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
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
        for repo_id in tqdm(repo_ids, desc="Processing datasets metadata"):
            try:
                ds_meta = LeRobotDatasetMetadata(
                    repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
                )
                delta_timestamps_dict[repo_id] = resolve_delta_timestamps(cfg.policy, ds_meta)
            except Exception as e:
                print(f"Error processing dataset {repo_id}: {e}")
                continue

        dataset = MultiLeRobotDataset(
            repo_ids,
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
        if isinstance(dataset, LeRobotDataset):
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
