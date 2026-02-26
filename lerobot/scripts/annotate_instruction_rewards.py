#!/usr/bin/env python
"""Annotate LeRobot datasets with instruction rewards for advantage-weighted behavior cloning.

This script computes instruction rewards using Qwen3-VL for each trajectory prefix,
then derives TD-style advantages and stores them directly in the dataset using modify_features().

The advantage column is added directly to the parquet files, making it available
during training without any additional loading mechanism.

Usage:
    python lerobot/scripts/annotate_instruction_rewards.py \
        --input_repo_id sengi/stack_red_cube_on_green_cube_adv \
        --output_dir ./datasets_out/ \
        --output_repo_id sengi/stack_red_cube_on_green_cube_adv \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --push_to_hub \
        --reward_stride 50 \
        --dry_run
"""

import argparse
from hmac import new
from pathlib import Path
from sys import prefix

import numpy as np
from loguru import logger
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset_v3 import LeRobotDatasetV3
from lerobot.common.datasets.v3.dataset_tools import add_features, modify_features as modify_features_v3
from lerobot.common.datasets.v21.dataset_tools import modify_features as modify_features_v21
from opengvl.clients.qwen import QwenClient
from opengvl.utils.logging_config import setup_logging
from opengvl.utils.metrics import spearman_dense_correlation
from lerobot.common.constants import HF_LEROBOT_HOME
import time

def parse_args():
    # Configure logging format
    setup_logging(level="INFO", format_type="detailed")
    parser = argparse.ArgumentParser(
        description="Annotate LeRobot datasets with instruction rewards stored directly in parquet"
    )
    parser.add_argument(
        "--input_repo_id",
        type=str,
        required=True,
        help="Source dataset repo_id (e.g., 'lerobot/pusht')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/",
        help="Directory to save the annotated dataset",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Output dataset repo_id (default: {input_repo_id}_with_advantages)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen VLM model to use for computing instruction rewards",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to annotate (default: all)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS for video input to the VLM (default: 2.0)",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Reduction method for log probabilities (default: mean)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sample every N-th frame for VLM evaluation (default: 1 = all frames)",
    )
    parser.add_argument(
        "--reward_stride",
        type=int,
        default=50,
        help="Compute instruction rewards every N sampled frames (default: 1 = every frame).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip reward computation; use constant advantages (20.0) but still write outputs.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the annotated dataset to the Hugging Face Hub using dataset.push_to_hub().",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision name for loading the input dataset (default: main).",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Create the hub dataset as private (defaults to public).",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="Root directory for input dataset (default: HF cache)",
    )
    parser.add_argument(
        "--force_cache_sync",
        action="store_true",
        help="Force re-syncing the dataset cache from the hub.",
    )
    return parser.parse_args()


def extract_frames_from_episode(dataset, episode_index: int, sample_interval: int = 1, dry_run: bool = False):
    """Extract frames using batch video decoding.

    Args:
        dataset: LeRobotDataset or LeRobotDatasetV3 instance.
        episode_index: Episode index to extract frames from.
        sample_interval: Sample every N-th frame for VLM evaluation.

    Returns:
        Tuple of (frames_list, instruction_text, frame_indices, total_frames, fps)
    """
    from lerobot.common.datasets.lerobot_dataset_v3 import LeRobotDatasetV3
    from lerobot.common.datasets.video_utils import decode_video_frames

    is_v3 = isinstance(dataset, LeRobotDatasetV3)

    # Get episode boundaries
    if is_v3:
        ep_data = dataset.meta.episodes[episode_index]
        start_idx = ep_data.get("dataset_from_index", 0)
        end_idx = ep_data.get("dataset_to_index", start_idx + ep_data.get("length", 0))
    else:
        start_idx = dataset.episode_data_index["from"][episode_index]
        end_idx = dataset.episode_data_index["to"][episode_index]

    total_frames = end_idx - start_idx
    if dry_run:
        return [], "", [], total_frames, None
    sampled_indices = list(range(start_idx, end_idx, sample_interval))

    # Get video key
    video_keys = list(dataset.meta.video_keys)
    if not video_keys:
        raise ValueError("No video keys found in dataset")
    # camera_key = video_keys[0]

    all_views = []
    video_keys.pop(-1)
    assert "wrist" not in video_keys[0]
    for camera_key in video_keys:
        # Batch-fetch timestamps from parquet
        if is_v3:
            dataset._ensure_hf_dataset_loaded()
        timestamps = [dataset.hf_dataset["timestamp"][idx].item() for idx in sampled_indices]

        # Get video path
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, camera_key)

        # V3 needs timestamp offset, V2.1 doesn't
        if is_v3:
            from_timestamp = ep_data[f"videos/{camera_key}/from_timestamp"]
            shifted_timestamps = [from_timestamp + ts for ts in timestamps]
        else:
            shifted_timestamps = timestamps

        # BATCH DECODE all frames at once
        frames_tensor = decode_video_frames(
            video_path, shifted_timestamps, dataset.tolerance_s, dataset.video_backend
        )

        # Convert to numpy HWC uint8
        frames = []
        for frame in frames_tensor:
            frame_np = frame.numpy()
            if frame_np.shape[0] in [1, 3]:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            frames.append(frame_np)

        all_views.append(frames)
        
    # Get instruction
    if is_v3:
        task_idx = dataset.hf_dataset["task_index"][start_idx].item()
        instruction = dataset.meta.tasks.iloc[task_idx].name
    else:
        task_idx = dataset.hf_dataset["task_index"][start_idx].item()
        instruction = dataset.meta.tasks[task_idx]

    fps = None
    if hasattr(dataset, "meta") and getattr(dataset.meta, "info", None):
        fps = dataset.meta.info.get("fps")
    elif hasattr(dataset, "info"):
        fps = dataset.info.get("fps")

    return all_views, instruction, sampled_indices, total_frames, fps


def compute_advantages_for_episode(
    client,
    frames: list,
    instruction: str,
    total_frames: int,
    sampled_indices: list,
    start_idx: int,
    fps: float = 2.0,
    reduction: str = "mean",
    reward_stride: int = 50,
) -> tuple[np.ndarray, float]:
    """Compute instruction rewards and TD-style advantages for an episode.

    Args:
        client: QwenClient instance.
        frames: List of frame images (numpy arrays in HWC format, uint8).
        instruction: Task instruction text.
        total_frames: Total number of frames in the episode.
        sampled_indices: Global indices of the sampled frames.
        start_idx: Starting global index of the episode.
        fps: FPS for video input.
        reduction: Reduction method for log probabilities.
        reward_stride: Compute rewards every N sampled frames (>=1), then broadcast.

    Returns:
        tuple: (advantages for ALL frames in the episode, VOC score over prefix rewards).
    """
    num_cam_views = len(frames)
    total_frames = len(frames[0])

    prefix_rewards = []

    # Compute reward once per chunk boundary (sampled frame) with optional stride.
    prefix_lengths = list(range(1, total_frames, max(reward_stride, 1)))
    if prefix_lengths[-1] != total_frames:
        prefix_lengths.append(total_frames)

    for t in tqdm(prefix_lengths, desc="Computing chunk rewards"):
        logger.info(f"Computing reward for prefix length {t}")
        prefix_frames = [view[:t] for view in frames]

        result = client.compute_instruction_reward(
            frames=prefix_frames,
            instruction=instruction,
            reduction=reduction,
            fps=fps,
            use_video_description=False,
            add_chat_template=True,
        )
        prefix_rewards.append(result.reward)
        logger.info(f"Prefix length {t}: reward = {result.reward:.4f}")

    if not prefix_rewards:
        return np.zeros(total_frames, dtype=np.float32), float("nan")

    voc_score = spearman_dense_correlation(prefix_rewards)

    prefix_rewards = np.array(prefix_rewards, dtype=np.float32)
    prefix_rewards = prefix_rewards - np.min(prefix_rewards)
    max_reward = float(np.max(prefix_rewards))
    if max_reward > 1e-8:
        prefix_rewards = prefix_rewards / max_reward
    else:
        # Flat rewards: keep neutral normalized rewards to avoid NaN.
        prefix_rewards = np.zeros_like(prefix_rewards, dtype=np.float32)
    tau = 2.0
    sampled_advantages = []
    if len(prefix_rewards) == 1:
        sampled_advantages = [1.0] * total_frames
    else:
        for i in range(1, len(prefix_rewards)):
            delta_reward = float(prefix_rewards[i] - prefix_rewards[i - 1])
            weight = float(np.exp(delta_reward * tau))
            for _ in range(prefix_lengths[i - 1], prefix_lengths[i]):
                sampled_advantages.append(weight)
        placeholder_advantage = sampled_advantages[0] if sampled_advantages else 1.0
        sampled_advantages = [placeholder_advantage] * prefix_lengths[0] + sampled_advantages

    # Now interpolate advantages to all frames in the episode
    # Each sampled frame's advantage applies to itself and the frames until the next sample
    all_advantages = np.zeros(total_frames, dtype=np.float32)

    # Convert sampled_indices to local indices within episode
    local_sampled_indices = [idx - start_idx for idx in sampled_indices]

    for i, local_idx in enumerate(local_sampled_indices):
        adv = sampled_advantages[i] if i < len(sampled_advantages) else sampled_advantages[-1]
        # Determine the range this advantage applies to
        end_local_idx = local_sampled_indices[i + 1] if i < len(local_sampled_indices) - 1 else total_frames

        # Assign the advantage to all frames in this range
        all_advantages[local_idx:end_local_idx] = adv

    return all_advantages, voc_score


def annotate_dataset(
    input_repo_id: str,
    output_dir: Path,
    output_repo_id: str,
    model_name: str,
    max_episodes: int | None = None,
    fps: float = 2.0,
    reduction: str = "mean",
    sample_interval: int = 1,
    reward_stride: int = 50,
    input_root: str | None = None,
    push_to_hub: bool = False,
    revision: str | None = None,
    hub_private: bool = False,
    force_cache_sync: bool = False,
    dry_run: bool = False,
):
    """Annotate a LeRobot dataset with instruction rewards stored in parquet."""

    # Load dataset
    logger.info(f"Loading dataset: {input_repo_id}")
    try:
        dataset = LeRobotDatasetV3(input_repo_id, root=input_root, force_cache_sync=force_cache_sync, revision=revision)
        is_v3 = True
    except Exception:
        logger.info("V3 loader failed, trying V2.1 loader...")
        dataset = LeRobotDataset(input_repo_id, root=input_root, force_cache_sync=force_cache_sync, revision=revision)
        is_v3 = False
    num_episodes = (
        dataset.meta.total_episodes if hasattr(dataset.meta, "total_episodes") else dataset.num_episodes
    )
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    total_frames = dataset.meta.total_frames if hasattr(dataset.meta, "total_frames") else len(dataset)
    logger.info(f"Dataset loaded: {num_episodes} episodes, {total_frames} frames to annotate")

    if not dry_run:
        # Initialize VLM client
        logger.info(f"Loading VLM: {model_name}")
        client = QwenClient(model_name=model_name)
    else:
        client = None

    # Compute advantages for all episodes
    all_advantages = []

    for ep_idx in range(num_episodes):
        logger.info(f"Annotating episode {ep_idx + 1}/{num_episodes}")
        # try:
        # Extract frames

        if dry_run:
            frames, instruction, sampled_indices, ep_total_frames, ep_fps = extract_frames_from_episode(
                dataset, ep_idx, sample_interval, dry_run=True
            )
            logger.warning(f"Episode {ep_idx} dry-run or fewer than 2 frames; skipping reward computation.")
            ep_advantages = (
                np.full(ep_total_frames, 1.0, dtype=np.float32)
            )
            voc_score = float("nan")
        else:
            frames, instruction, sampled_indices, ep_total_frames, ep_fps = extract_frames_from_episode(
                dataset, ep_idx, sample_interval
            )
            # instruction = "pick up the doll and lift it up, then move it to the box and put it inside the box."
            fps_used = ep_fps if ep_fps is not None else fps
            logger.info(f"Episode {ep_idx}: {ep_total_frames} frames, instruction: '{instruction[:50]}...'")

            # Get episode start index
            if is_v3:
                start_idx = dataset.meta.episodes[ep_idx].get("dataset_from_index", 0)
            else:
                start_idx = dataset.episode_data_index["from"].get(ep_idx, 0)

            # Compute advantages and per-episode VOC
            ep_advantages, voc_score = compute_advantages_for_episode(
                client,
                frames,
                instruction,
                ep_total_frames,
                sampled_indices,
                start_idx,
                fps=fps_used,
                reduction=reduction,
                reward_stride=reward_stride,
            )
            assert len(ep_advantages) == ep_total_frames, f"Advantage length mismatch, {len(ep_advantages)} vs {ep_total_frames}"

        # Log advantage statistics for visibility
        logger.info(
            "Episode {} advantage stats — min: {:.4f}, max: {:.4f}, mean: {:.4f}, voc: {:.4f}".format(
                ep_idx,
                float(ep_advantages.min()),
                float(ep_advantages.max()),
                float(ep_advantages.mean()),
                float(voc_score),
            )
        )

        all_advantages.append(ep_advantages)

    # Concatenate all advantages into a single array
    advantages_array = np.concatenate(all_advantages)
    logger.info(f"Computed {len(advantages_array)} advantages")
    logger.info(
        f"Advantage stats: mean={advantages_array.mean():.4f}, std={advantages_array.std():.4f}, "
        f"min={advantages_array.min():.4f}, max={advantages_array.max():.4f}"
    )

    if dry_run and push_to_hub:
        logger.info("Dry run enabled: proceeding with hub upload.")

    # Use modify_features to add the advantage column
    logger.info(f"Creating new dataset with advantage column at {output_dir / output_repo_id}")

    # Select appropriate modify_features based on dataset version
    modify_features = modify_features_v3 if is_v3 else modify_features_v21

    target_root = (output_dir / output_repo_id).resolve()
    source_root = Path(dataset.root).resolve()
    if target_root == source_root:
        raise ValueError(
            f"Output path {target_root} matches input dataset root. "
            "Use a different --output_dir or --output_repo_id to avoid overwriting."
        )

    if target_root.exists():
        raise ValueError(
            f"Output path {target_root} already exists. "
            "Use a different --output_dir or --output_repo_id to avoid overwriting."
        )
    add_features = {"advantage": (advantages_array, {"dtype": "float32", "shape": (1,), "names": None})}
    new_dataset = modify_features(
        dataset=dataset,
        add_features=add_features,
        output_dir=output_dir / output_repo_id,
        repo_id=output_repo_id,
    )
    # new_dataset = dataset

    logger.info(f"Dataset created: {new_dataset.repo_id}")
    logger.info(f"Features: {list(new_dataset.meta.features.keys())}")
    logger.info(f"'advantage' in features: {'advantage' in new_dataset.meta.features}")

    # Verify by loading a sample
    sample = new_dataset[0]
    if "advantage" in sample:
        logger.info(f"Sample advantage value: {sample['advantage']}")
    else:
        logger.warning("'advantage' not found in sample!")

    if push_to_hub:
        new_dataset.push_to_hub(
            private=hub_private,
            push_videos=True,
            tag_version=True,
        )
        logger.info(f"Pushed dataset to hub: {new_dataset.repo_id}")

    return new_dataset


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output repo_id
    if args.output_repo_id is None:
        # Strip any path prefix and add suffix
        base_name = args.input_repo_id.split("/")[-1]
        output_repo_id = f"{base_name}_with_advantages"
    else:
        output_repo_id = args.output_repo_id

    logger.info(f"Processing dataset: {args.input_repo_id}")
    logger.info(f"Output: {output_dir / output_repo_id}")

    # Early guard: avoid writing into the same directory as the source dataset.

    output_path = (output_dir / output_repo_id).resolve()
    candidate_roots = []
    if args.input_root is not None:
        candidate_roots.append((Path(args.input_root) / args.input_repo_id).resolve())
        candidate_roots.append(Path(args.input_root).resolve())
    candidate_roots.append((HF_LEROBOT_HOME / args.input_repo_id).resolve())
    if not args.dry_run and output_path in candidate_roots:
        raise ValueError(
            f"Output path {output_path} matches an input dataset root. "
            "Choose a different --output_dir or --output_repo_id."
        )

    annotate_dataset(
        input_repo_id=args.input_repo_id,
        input_root=args.input_root,
        output_dir=output_dir,
        output_repo_id=output_repo_id,
        model_name=args.model_name,
        max_episodes=args.max_episodes,
        fps=args.fps,
        reduction=args.reduction,
        sample_interval=args.sample_interval,
        reward_stride=args.reward_stride,
        push_to_hub=args.push_to_hub,
        revision=args.revision,
        hub_private=args.hub_private,
        dry_run=args.dry_run,
        force_cache_sync=args.force_cache_sync,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    main()
