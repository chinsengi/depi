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

import logging
import time
import warnings
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from lerobot.common.constants import CHECKPOINTS_DIR
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import close_envs
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    cleanup_old_checkpoints,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_best_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use the new training script `accelerate_train.py` instead.",
    DeprecationWarning,
    stacklevel=2,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    params: list[torch.Tensor],
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    gradient_accumulation_steps: int = 1,
    step: int = 0,
    loss_threshold: float = 0.04,
) -> tuple[MetricsTracker, dict]:
    device = get_device_from_parameters(policy)
    policy.train()
    start_time = time.perf_counter()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    train_metrics.forward_s = time.perf_counter() - start_time

    # Scale loss by gradient accumulation steps
    loss = loss / gradient_accumulation_steps

    # we don't want to backpropagate if the loss is too high later in the training, since that may due to the data quality
    if output_dict["l2_loss"] < loss_threshold or step < 10000:
        grad_scaler.scale(loss).backward()

    train_metrics.bkw_s = time.perf_counter() - start_time - train_metrics.forward_s.val
    # Gradient accumulation
    if step % gradient_accumulation_steps == 0:
        # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
        grad_scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            grad_clip_norm,
            error_if_nonfinite=True,
            foreach=True,
        )

        # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        with lock if lock is not None else nullcontext():
            grad_scaler.step(optimizer)
        # Updates the scale for next iteration.
        grad_scaler.update()

        optimizer.zero_grad()
        # time used to update the parameters
        train_metrics.update_s = (
            time.perf_counter() - start_time - train_metrics.forward_s.val - train_metrics.bkw_s.val
        )
        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

        if has_method(policy, "update"):
            # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
            policy.update()

        train_metrics.loss = loss.item() * gradient_accumulation_steps  # Scale loss back for logging
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
    else:
        train_metrics.loss = loss.item() * gradient_accumulation_steps  # Scale loss back for logging
        # train_metrics.grad_norm = 0.0  # No gradient norm when not updating
        train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_best_torch_device()
    cfg.policy.device = str(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    logging.info("Created dataset")

    # enable wandb logging after dataset is created
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = StatefulDataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # create policy
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, compile=cfg.compile, strict=cfg.strict)
    if policy.model is not None and isinstance(policy.model, torch._dynamo.OptimizedModule):
        logging.info("Policy created with compiled model")
        cfg.compile = True

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    logging.info("Optimizer and scheduler created")

    step = 0  # number of policy updates (forward + backward + optim)
    if cfg.resume:
        try:
            if cfg.resume_scheduler:
                step, optimizer, lr_scheduler = load_training_state(
                    cfg.checkpoint_path, optimizer, lr_scheduler, dataloader
                )
            else:
                step, optimizer, _ = load_training_state(cfg.checkpoint_path, optimizer, None, dataloader)
        except Exception as e:
            logging.info(f"resume error: {e}")
            raise e

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "forward_s": AverageMeter("fwd_s", ":.3f"),
        "bkw_s": AverageMeter("bkw_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    logging.info("Start offline training on a fixed dataset")
    params = list(policy.parameters())
    for _, batch in enumerate(dataloader):
        if step >= cfg.steps:
            break
        else:
            step += 1
        logging.info(f"step: {step}")

        if cfg.test_dataloader:
            logging.info(f"data idxs: {batch['index']}")
            continue

        start_time = time.perf_counter()
        train_tracker.dataloading_s = time.perf_counter() - start_time
        if train_tracker.dataloading_s.val > 0.02 and isinstance(dataset, MultiLeRobotDataset):
            print("\n")
            logging.warning(
                f"dataloading takes too long:dataloading_s: {train_tracker.dataloading_s.val}, dataset: {[dataset.repo_index_to_id[idx.item()] for idx in batch['dataset_index']]}, dataset_index: {batch['dataset_index']}"
            )
            logging.warning(f"episode_index:{batch['episode_index']}")

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        if step % cfg.gradient_accumulation_steps == 0 and cfg.compile:
            torch.compiler.cudagraph_mark_step_begin()

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            params,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            step=step,
            loss_threshold=cfg.loss_threshold,
        )
        if output_dict["l2_loss"] > cfg.loss_threshold and step > 10000:
            logging.warning(f"Step {step} | loss too high: \n l2_loss: {output_dict['l2_loss']}")
            if isinstance(dataset, MultiLeRobotDataset):
                idx_to_id = dataset.repo_index_to_id
                logging.warning(f"dataset_id:{[idx_to_id[idx.item()] for idx in batch['dataset_index']]}")
            logging.warning(f"episode_index:{batch['episode_index']}")
            output_dict["l2_loss"] = 0.0

        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            print("\n")
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            logging.info(f"Checkpoint policy after step {step} to {checkpoint_dir}")
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, dataloader)

            update_last_checkpoint(checkpoint_dir)
            # Clean up old checkpoints, keeping only the last 2
            cleanup_old_checkpoints(cfg.output_dir / CHECKPOINTS_DIR, keep_last_n=2)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size,
                dataset.num_frames,
                dataset.num_episodes,
                eval_metrics,
                initial_step=step,
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                video_paths = eval_info.get("video_paths", [])
                if video_paths:
                    wandb_logger.log_video(video_paths[0], step, mode="eval")
    if eval_env:
        close_envs(eval_env)
    logging.info(f"End of training, total steps: {step}")


if __name__ == "__main__":
    init_logging()
    train()
