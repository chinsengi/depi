# Copyright 2025 Shirui Chen
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
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, TorchDynamoPlugin
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lerobot.common.constants import CHECKPOINTS_DIR
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import close_envs
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    cleanup_old_checkpoints,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_step,
    reconcile_checkpoint_compile_state,
    save_training_step,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    step: int = 0,
    loss_threshold: float = 0.04,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()

    loss, output_dict = policy.forward(batch)
    # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    loss = loss.mean()
    train_metrics.forward_s = time.perf_counter() - start_time

    # we don't want to backpropagate if the loss is too high later in the training, since that may due to the data quality
    # if output_dict["l2_loss"] < loss_threshold or step < 10000:
    # Use accelerator's backward method
    accelerator.backward(loss)

    train_metrics.bkw_s = time.perf_counter() - start_time - train_metrics.forward_s.val

    # Following accelerator's recommended pattern:
    # Always call optimizer.step() and optimizer.zero_grad() on every iteration
    # The accelerator context manager handles gradient accumulation automatically

    # Get parameters for gradient clipping
    params = list(policy.parameters())

    # Clip gradients using accelerator's method
    grad_norm = accelerator.clip_grad_norm_(params, grad_clip_norm)

    # Optimizer step using accelerator
    optimizer.step()
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    # time used to update the parameters
    train_metrics.update_s = (
        time.perf_counter() - start_time - train_metrics.forward_s.val - train_metrics.bkw_s.val
    )

    train_metrics.loss = loss.item()  # No need to scale back for logging
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = (
        lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]
    )

    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Initialise Accelerator – handles multi-GPU / multi-node & mixed precision
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.ddp_timeout_s))
    dynamo_plugin = TorchDynamoPlugin(
        use_regional_compilation=False,
        backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        mode="default",  # Options: "default", "reduce-overhead", "max-autotune"
        disable=not cfg.compile,
    )
    dataloader_cfg = DataLoaderConfiguration(
        non_blocking=True,
        use_stateful_dataloader=False,  # TODO: there are right now a lot of issues with stateful dataloader of accelerate, so we disable it for now
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
        dataloader_config=dataloader_cfg,
    )

    # Reinitialize logging to only log on main process
    init_logging(accelerator, only_main=True)
    logging.info(pformat(cfg.to_dict()))

    device = accelerator.device
    if cfg.policy is not None:
        cfg.policy.device = str(device)
    # No effect on performance whatsoever
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if accelerator.is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()
    logging.info("Dataset created by main process")

    # Now all other processes can safely load the dataset
    if not accelerator.is_main_process:
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    logging.info("Dataset created")

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and accelerator.is_main_process:
        logging.info("Creating env")
        eval_env = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
        )

    # enable wandb logging after dataset is created
    if cfg.wandb.enable and cfg.wandb.project and accelerator.is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if accelerator.is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

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

    dataloader = DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    logging.info("dataloader instantiated")

    # create policy
    logging.info("Creating policy")
    accelerator.wait_for_everyone()
    # when use accelerate, we compile the policy through accelerate config
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, compile=False, strict=cfg.strict)
    accelerator.wait_for_everyone()
    logging.info("Creating optimizer and scheduler")
    # https://huggingface.co/docs/accelerate/concept_guides/performance
    # cfg.scheduler.num_warmup_steps *= accelerator.num_processes
    # cfg.scheduler.num_decay_steps *= accelerator.num_processes
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    logging.info("Optimizer and scheduler created")

    # from lerobot.common.datasets.utils import cycle
    # dl_iter = cycle(dataloader)
    # try:
    #     _ = next(dl_iter)
    # except Exception as e:
    #     logging.info(f"dataloader error after accelerator.prepare: {e}")
    #     return

    step = 0  # number of policy updates (forward + backward + optim)

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
    # Note that accelerator prepare will skip compiling the policy if it is already compiled
    policy.model, optimizer, dataloader = accelerator.prepare(policy.model, optimizer, dataloader)
    lr_scheduler: AcceleratedScheduler = accelerator.prepare(lr_scheduler)
    lr_scheduler.step_with_optimizer = False  # so that the lr scheduler is only called once per process.
    policy.to(device)

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
        cfg.batch_size * accelerator.num_processes,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )
    accelerator.register_for_checkpointing(train_tracker)

    # Resume training after accelerator.prepare() if needed
    if cfg.resume:
        reconcile_checkpoint_compile_state(Path(cfg.checkpoint_path), policy.model)
        accelerator.load_state(cfg.checkpoint_path, strict=True)
        # Get the step from the loaded state
        step = load_training_step(cfg.checkpoint_path) + 1
        accelerator.wait_for_everyone()

    skipped_dataloader = accelerator.skip_first_batches(
        dataloader, ((train_tracker.samples // cfg.batch_size) // accelerator.num_processes) % len(dataloader)
    )

    logging.info("Start offline training on a fixed dataset")
    # Track actual optimization steps (not batch steps)
    init_step = step  # Initialize from resume step if resuming
    current_opt_step = init_step
    for epoch in range(cfg.num_epochs):
        if current_opt_step > cfg.steps:
            break
        init_step = current_opt_step
        current_dataloader = skipped_dataloader if epoch == 0 else dataloader

        for batch_idx, batch in enumerate(current_dataloader):
            # Accelerator handles gradient accumulation, so we track actual optimization steps
            # Each batch corresponds to one forward pass, but optimization happens every gradient_accumulation_steps
            current_opt_step = init_step + (batch_idx // cfg.gradient_accumulation_steps)

            if current_opt_step > cfg.steps:
                break

            if cfg.test_dataloader:
                logging.info(f"data idxs: {batch['index']}")
                continue

            start_time = time.perf_counter()
            train_tracker.dataloading_s = time.perf_counter() - start_time
            # if train_tracker.dataloading_s.val > 0.02 and isinstance(dataset, MultiLeRobotDataset):
            #     logging.warning(
            #         f"dataloading takes too long:dataloading_s: {train_tracker.dataloading_s.val}, dataset: {[dataset.repo_index_to_id[idx.item()] for idx in batch['dataset_index']]}, dataset_index: {batch['dataset_index']}"
            #     )
            #     logging.warning(f"episode_index:{batch['episode_index']}")

            # Use accelerator's gradient accumulation context
            with accelerator.accumulate(policy):
                train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                    step=current_opt_step,
                    loss_threshold=cfg.loss_threshold,
                )

            # if output_dict["l2_loss"] > cfg.loss_threshold and current_opt_step > 10000:
            #     logging.warning(f"Step {current_opt_step} | loss too high: \n l2_loss: {output_dict['l2_loss']}")
            #     if isinstance(dataset, MultiLeRobotDataset):
            #         idx_to_id = dataset.repo_index_to_id
            #         logging.warning(f"dataset_id:{[idx_to_id[idx.item()] for idx in batch['dataset_index']]}")
            #     logging.warning(f"episode_index:{batch['episode_index']}")
            #     output_dict["l2_loss"] = 0.0

            train_tracker.step()
            is_log_step = is_saving_step = is_eval_step = False
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                global_opt_step = current_opt_step + 1
                is_log_step = cfg.log_freq > 0 and global_opt_step % cfg.log_freq == 0
                is_saving_step = (
                    cfg.save_freq > 0 and global_opt_step % cfg.save_freq == 0
                ) or global_opt_step == cfg.steps
                is_eval_step = cfg.eval_freq > 0 and global_opt_step % cfg.eval_freq == 0

            if is_log_step:
                logging.info(f"batch_idx: {batch_idx}, opt_step: {global_opt_step}")
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, global_opt_step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step and accelerator.is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, global_opt_step)

                # Use accelerator's save method
                logging.info(f"Checkpoint policy after step {global_opt_step} to {checkpoint_dir}")
                accelerator.save_state(checkpoint_dir)
                cfg.save_pretrained(checkpoint_dir / "configs")
                save_training_step(global_opt_step, checkpoint_dir)

                update_last_checkpoint(checkpoint_dir)
                # Clean up old checkpoints, keeping only the last 2
                # TODO: make the number of kept checkpoints a config param
                cleanup_old_checkpoints(cfg.output_dir / CHECKPOINTS_DIR, keep_last_n=2)

                unwrapped_model = accelerator.unwrap_model(policy.model, keep_torch_compile=False)
                model = policy.model
                policy.model = unwrapped_model
                policy.save_pretrained(checkpoint_dir / "pretrained_model")
                policy.model = model
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.save_checkpoint and is_saving_step:
                accelerator.wait_for_everyone()

            if is_eval_step:
                if accelerator.is_main_process and eval_env is not None:
                    step_id = get_step_identifier(global_opt_step, cfg.steps)
                    logging.info(f"Eval policy at step {global_opt_step}")
                    unwrapped_model = accelerator.unwrap_model(policy.model, keep_torch_compile=False)
                    wrapped_model = policy.model
                    policy.model = unwrapped_model
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
                    policy.model = wrapped_model

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
                        initial_step=global_opt_step,
                    )
                    eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                    eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                    eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                    logging.info(eval_tracker)
                    if wandb_logger:
                        wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                        wandb_logger.log_dict(wandb_log_dict, global_opt_step, mode="eval")
                        video_paths = eval_info.get("video_paths", [])
                        if video_paths:
                            wandb_logger.log_video(video_paths[0], global_opt_step, mode="eval")
                accelerator.wait_for_everyone()

            end_time = time.perf_counter()
            logging.info(f"Time taken for batch {batch_idx}: {end_time - start_time:.2f} seconds")

    policy.model = accelerator.unwrap_model(policy.model)
    policy.save_pretrained(Path(cfg.output_dir) / "pretrained_model")
    if eval_env is not None and accelerator.is_main_process:
        close_envs(eval_env)
    logging.info(f"End of training, total steps: {train_tracker.steps}")


if __name__ == "__main__":
    train()
