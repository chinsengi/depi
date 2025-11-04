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
#
# Modified by Shirui Chen in 2025
# Based on code originally licensed under Apache License 2.0

import contextlib
import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.utils
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.datasets.backward_compatibility import BackwardCompatibilityError
from lerobot.common.datasets._shared_mixins import (
    DatasetCommonMixin,
    DatasetMetadataAccessorsMixin,
)
from lerobot.common.datasets.v3 import utils as v3_utils
from lerobot.common.datasets.v3.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.v3.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_IMAGE_PATH,
    check_delta_timestamps,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_file_size_in_mb,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_nested_dataset,
    update_chunk_file_indices,
    validate_episode_buffer,
    validate_frame,
    write_info,
)
from lerobot.common.datasets.v3.video_utils import (
    VideoFrame,
    concatenate_video_files,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_duration_in_s,
    get_video_info,
)
from lerobot.common.utils.constants import HF_LEROBOT_HOME

CODEBASE_VERSION_V3 = "v3.0"


@dataclass
class LeRobotDatasetMetadataV3(DatasetMetadataAccessorsMixin):
    repo_id: str
    root: str | Path | None = None
    revision: str | None = None
    force_cache_sync: bool = False
    metadata_buffer_size: int = 10

    writer: pq.ParquetWriter | None = field(init=False, default=None)
    latest_episode: dict | None = field(init=False, default=None)
    metadata_buffer: list[dict] = field(init=False, default_factory=list)
    tasks: pd.DataFrame | None = field(init=False, default=None)
    episodes: list | None = field(init=False, default=None)
    stats: dict | None = field(init=False, default=None)
    info: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root) if self.root is not None else HF_LEROBOT_HOME / self.repo_id
        self.revision = self.revision if self.revision else CODEBASE_VERSION_V3

        try:
            if self.force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if v3_utils.is_valid_version(self.revision):
                self.revision = v3_utils.get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def _flush_metadata_buffer(self) -> None:
        """Write all buffered episode metadata to parquet file."""
        if not hasattr(self, "metadata_buffer") or len(self.metadata_buffer) == 0:
            return

        combined_dict: dict[str, list] = {}
        for episode_dict in self.metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []
                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        first_ep = self.metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        table = pa.Table.from_pydict(combined_dict)

        if not self.writer:
            path = Path(self.root / v3_utils.DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)

        self.writer.write_table(table)

        self.latest_episode = self.metadata_buffer[-1]
        self.metadata_buffer.clear()

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        self._flush_metadata_buffer()

        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self._close_writer()

    def load_metadata(self) -> None:
        self.info = v3_utils.load_info(self.root)
        v3_utils.check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION_V3)
        self.tasks = v3_utils.load_tasks(self.root)
        self.episodes = v3_utils.load_episodes(self.root)
        self.stats = v3_utils.load_stats(self.root)

    @property
    def url_root(self) -> str:
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        if self.episodes is None:
            self.episodes = v3_utils.load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )

        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if self.episodes is None:
            self.episodes = v3_utils.load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    @property
    def chunks_size(self) -> int:
        """Max number of files per chunk."""
        return self.info["chunks_size"]

    @property
    def data_files_size_in_mb(self) -> int:
        """Max size of data file in mega bytes."""
        return self.info["data_files_size_in_mb"]

    @property
    def video_files_size_in_mb(self) -> int:
        """Max size of video file in mega bytes."""
        return self.info["video_files_size_in_mb"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        if self.tasks is None:
            return None
        return int(self.tasks.loc[task].task_index) if task in self.tasks.index else None

    def save_episode_tasks(self, tasks: list[str]):
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=tasks)
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            v3_utils.write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """Buffer episode metadata and write to parquet in batches for efficiency.

        This function accumulates episode metadata in a buffer and flushes it when the buffer
        reaches the configured size. This reduces I/O overhead by writing multiple episodes
        at once instead of one row at a time.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert to list format for each value
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        num_frames = episode_dict["length"][0]

        if self.latest_episode is None:
            chunk_idx, file_idx = 0, 0
            if self.episodes is not None and len(self.episodes) > 0:
                chunk_idx = self.episodes[-1]["meta/episodes/chunk_index"]
                file_idx = self.episodes[-1]["meta/episodes/file_index"]
                latest_num_frames = self.episodes[-1]["dataset_to_index"]
                episode_dict["dataset_from_index"] = [latest_num_frames]
                episode_dict["dataset_to_index"] = [latest_num_frames + num_frames]
                chunk_idx, file_idx = v3_utils.update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
            else:
                episode_dict["dataset_from_index"] = [0]
                episode_dict["dataset_to_index"] = [num_frames]

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
        else:
            chunk_idx = self.latest_episode["meta/episodes/chunk_index"][0]
            file_idx = self.latest_episode["meta/episodes/file_index"][0]

            latest_path = (
                self.root / v3_utils.DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                if self.writer is None
                else self.writer.where
            )

            if Path(latest_path).exists():
                latest_size_in_mb = v3_utils.get_file_size_in_mb(Path(latest_path))
                latest_num_frames = self.latest_episode["episode_index"][0]

                av_size_per_frame = latest_size_in_mb / latest_num_frames if latest_num_frames > 0 else 0.0

                if latest_size_in_mb + av_size_per_frame * num_frames >= self.data_files_size_in_mb:
                    self._flush_metadata_buffer()
                    chunk_idx, file_idx = v3_utils.update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
                    self._close_writer()

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
            episode_dict["dataset_from_index"] = [self.latest_episode["dataset_to_index"][0]]
            episode_dict["dataset_to_index"] = [self.latest_episode["dataset_to_index"][0] + num_frames]

        self.metadata_buffer.append(episode_dict)
        self.latest_episode = episode_dict

        if len(self.metadata_buffer) >= self.metadata_buffer_size:
            self._flush_metadata_buffer()

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        self.episodes = v3_utils.load_episodes(self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(v3_utils.flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        v3_utils.write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        v3_utils.write_stats(self.stats, self.root)

    def update_video_info(self, video_key: str | None = None) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.video_path.format(video_key=key, chunk_index=0, file_index=0)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """Update chunk and file size settings after dataset creation.

        This allows users to customize storage organization without modifying the constructor.
        These settings control how episodes are chunked and how large files can grow before
        creating new ones.

        Args:
            chunks_size: Maximum number of files per chunk directory. If None, keeps current value.
            data_files_size_in_mb: Maximum size for data parquet files in MB. If None, keeps current value.
            video_files_size_in_mb: Maximum size for video files in MB. If None, keeps current value.
        """
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info["chunks_size"] = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info["data_files_size_in_mb"] = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info["video_files_size_in_mb"] = video_files_size_in_mb

        v3_utils.write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """Get current chunk and file size settings.

        Returns:
            Dict containing chunks_size, data_files_size_in_mb, and video_files_size_in_mb.
        """
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
        metadata_buffer_size: int = 10,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> "LeRobotDatasetMetadataV3":
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **v3_utils.DEFAULT_FEATURES}
        v3_utils._validate_feature_names(features)

        obj.tasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = v3_utils.create_empty_dataset_info(
            CODEBASE_VERSION_V3,
            fps,
            features,
            use_videos,
            robot_type,
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
        )
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError("Features include videos but use_videos is False")
        v3_utils.write_json(obj.info, obj.root / v3_utils.INFO_PATH)
        obj.revision = None
        obj.writer = None
        obj.latest_episode = None
        obj.metadata_buffer = []
        obj.metadata_buffer_size = metadata_buffer_size
        obj.force_cache_sync = False
        return obj


class LeRobotDatasetV3(DatasetCommonMixin, torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v3.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v2.1 to v3.0, which you can find at
              lerobot/datasets/v30/convert_dataset_v21_to_v30.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDatasetV3 with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDatasetV3 format.


        In terms of files, LeRobotDatasetV3 encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDatasetV3 looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. Multiple episodes are
        consolidated into chunked files which improves storage efficiency and loading performance. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            force_cache_sync (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION_V3
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None
        self.writer = None
        self.latest_episode = None
        self._current_file_start_frame = None  # Track the starting frame index of the current parquet file

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadataV3(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # Track dataset state for efficient incremental writing
        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            # Check if cached dataset contains all requested episodes
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

    def __del__(self):
        """
        Trust the user to call .finalize() but as an added safety check call the parquet writer to stop when calling the destructor
        """
        self._close_writer()

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION_V3, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION_V3, revision=branch, repo_type="dataset")

    def download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        # Get available episode indices from cached dataset
        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        # Determine requested episodes
        if self.episodes is None:
            # Requesting all episodes - check if we have all episodes from metadata
            requested_episodes = set(range(self.meta.total_episodes))
        else:
            # Requesting specific episodes
            requested_episodes = set(self.episodes)

        # Check if all requested episodes are available in cached data
        if not requested_episodes.issubset(available_episodes):
            return False

        # Check if all required video files exist
        if len(self.meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self.meta.video_keys:
                    video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes.

        Note: When episodes a subset of the full dataset is requested, we must return the
        actual loaded data length (len(self.hf_dataset)) rather than metadata total_frames.
        self.meta.total_frames is the total number of frames in the full dataset.
        """
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self.meta.total_frames

    @property
    def _image_path_template(self) -> str:
        return DEFAULT_IMAGE_PATH

    def _build_hf_features(self) -> datasets.Features:
        return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _ensure_hf_dataset_loaded(self):
        """Lazy load the HF dataset only when needed for reading."""
        if self._lazy_loading or self.hf_dataset is None:
            # Close the writer before loading to ensure parquet file is properly finalized
            if self.writer is not None:
                self._close_writer()
                self._writer_closed_for_reading = True
            self.hf_dataset = self.load_hf_dataset()
            self._lazy_loading = False

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def finalize(self):
        """
        Close the parquet writers. This function needs to be called after data collection/conversion, else footer metadata won't be written to the parquet files.
        The dataset won't be valid and can't be loaded as ds = LeRobotDatasetV3(repo_id=repo, root=HF_LEROBOT_HOME.joinpath(repo))
        """
        self._close_writer()
        self.meta._close_writer()

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))  # Remove task from frame after processing

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        """
        Batch save videos for multiple episodes.

        Args:
            start_episode: Starting episode index (inclusive)
            end_episode: Ending episode index (exclusive). If None, encodes all episodes from start_episode to the current episode.
        """
        if end_episode is None:
            end_episode = self.num_episodes

        logging.info(
            f"Batch encoding {self.batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self.meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self.meta.episodes[start_episode]["data/file_index"]
        episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logging.info(f"Encoding videos for episode {ep_idx}")

            if (
                self.meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self.meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                # The current episode is in a new chunk or file.
                # Save previous episode dataframe and update the Hugging Face dataset by reloading it.
                episode_df.to_parquet(episode_df_path)
                self.meta.episodes = load_episodes(self.root)

                # Load new episode dataframe
                chunk_idx = self.meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self.meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            # Save the current episode's video metadata to the dataframe
            video_ep_metadata = {}
            for video_key in self.meta.video_keys:
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )  # allows NaN values along with integers

            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            self.meta.episodes = load_episodes(self.root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """Save episode data to a parquet file and update the Hugging Face dataset of frames data.

        This function processes episodes data from a buffer, converts it into a Hugging Face dataset,
        and saves it as a parquet file. It handles both the creation of new parquet files and the
        updating of existing ones based on size constraints. After saving the data, it reloads
        the Hugging Face dataset to ensure it is up-to-date.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert buffer into HF Dataset
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_num_frames = len(ep_dataset)

        if self.latest_episode is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            global_frame_index = 0
            self._current_file_start_frame = 0
            # However, if the episodes already exists
            # It means we are resuming recording, so we need to load the latest episode
            # Update the indices to avoid overwriting the latest episode
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                latest_ep = self.meta.episodes[-1]
                global_frame_index = latest_ep["dataset_to_index"]
                chunk_idx = latest_ep["data/chunk_index"]
                file_idx = latest_ep["data/file_index"]

                # When resuming, move to the next file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._current_file_start_frame = global_frame_index
        else:
            # Retrieve information from the latest parquet file
            latest_ep = self.latest_episode
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]
            global_frame_index = latest_ep["index"][-1] + 1

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_file_size_in_mb(latest_path)

            frames_in_current_file = global_frame_index - self._current_file_start_frame
            av_size_per_frame = (
                latest_size_in_mb / frames_in_current_file if frames_in_current_file > 0 else 0
            )

            # Determine if a new parquet file is needed
            if (
                latest_size_in_mb + av_size_per_frame * ep_num_frames >= self.meta.data_files_size_in_mb
                or self._writer_closed_for_reading
            ):
                # Size limit is reached or writer was closed for reading, prepare new parquet file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._close_writer()
                self._writer_closed_for_reading = False
                self._current_file_start_frame = global_frame_index

        ep_dict["data/chunk_index"] = chunk_idx
        ep_dict["data/file_index"] = file_idx

        # Write the resulting dataframe from RAM to disk
        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = ep_dataset.with_format("arrow")[:]
        if not self.writer:
            self.writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        self.writer.write_table(table)

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_index,
            "dataset_to_index": global_frame_index + ep_num_frames,
        }

        # Store metadata with episode data for next episode
        self.latest_episode = {**ep_dict, **metadata}

        # Mark that the HF dataset needs reloading (lazy loading approach)
        # This avoids expensive reloading during sequential recording
        self._lazy_loading = True
        # Update recorded frames count for efficient length tracking
        self._recorded_frames += ep_num_frames

        return metadata

    def _save_episode_video(self, video_key: str, episode_index: int) -> dict:
        # Encode episode frames into a temporary video
        ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if (
            episode_index == 0
            or self.meta.latest_episode is None
            or f"videos/{video_key}/chunk_index" not in self.meta.latest_episode
        ):
            # Initialize indices for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                # It means we are resuming recording, so we need to load the latest episode
                # Update the indices to avoid overwriting the latest episode
                old_chunk_idx = self.meta.episodes[-1][f"videos/{video_key}/chunk_index"]
                old_file_idx = self.meta.episodes[-1][f"videos/{video_key}/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(
                    old_chunk_idx, old_file_idx, self.meta.chunks_size
                )
            latest_duration_in_s = 0.0
            new_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            # Retrieve information from the latest updated video file using latest_episode
            latest_ep = self.meta.latest_episode
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"][0]
            file_idx = latest_ep[f"videos/{video_key}/file_index"][0]

            latest_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_file_size_in_mb(latest_path)
            latest_duration_in_s = latest_ep[f"videos/{video_key}/to_timestamp"][0]

            if latest_size_in_mb + ep_size_in_mb >= self.meta.video_files_size_in_mb:
                # Move temporary episode video to a new video file in the dataset
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                new_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                # Update latest video file
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # Remove temporary directory
        shutil.rmtree(str(ep_path.parent))

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if episode_index == 0:
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        # Clean up image files for the current episode buffer
        if delete_images:
            # Wait for the async image writer to finish
            if self.image_writer is not None:
                self._wait_image_writer()
            episode_index = self.episode_buffer["episode_index"]
            if isinstance(episode_index, np.ndarray):
                episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_video_frames(img_dir, temp_path, self.fps, overwrite=True)
        shutil.rmtree(img_dir)
        return temp_path

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> "LeRobotDatasetV3":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadataV3.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.writer = None
        obj.latest_episode = None
        obj._current_file_start_frame = None
        # Initialize tracking for incremental recording
        obj._lazy_loading = False
        obj._recorded_frames = 0
        obj._writer_closed_for_reading = False
        return obj


def _load_dataset_version_aware(
    repo_id: str,
    *,
    root: Path,
    episodes: list[int] | None,
    image_transforms: Callable | None,
    delta_timestamps: dict[str, list[float]] | None,
    tolerance_s: float,
    download_videos: bool,
    video_backend: str | None,
    force_cache_sync: bool = False,
    use_annotated_tasks: bool = False,
) -> torch.utils.data.Dataset:
    """
    Instantiate a LeRobot dataset, automatically selecting the implementation that
    matches the dataset's codebase version.
    """
    try:
        return LeRobotDatasetV3(
            repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            video_backend=video_backend,
            force_cache_sync=force_cache_sync,
        )
    except BackwardCompatibilityError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetV2

        logging.info(
            "Detected v2.x dataset format for %s; falling back to LeRobotDatasetV2.", repo_id
        )
        return LeRobotDatasetV2(
            repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            video_backend=video_backend,
            force_cache_sync=force_cache_sync,
            use_annotated_tasks=use_annotated_tasks,
        )


from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset as _UnifiedMultiLeRobotDataset

MultiLeRobotDataset = _UnifiedMultiLeRobotDataset
MultiLeRobotDatasetV3 = _UnifiedMultiLeRobotDataset
# Backward-compatible alias to preserve the previous public class name.
LeRobotDataset = LeRobotDatasetV3
