#!/bin/bash

n_gpus=1

# Use local HF cache only (repos removed from Hub)
# export HF_HUB_OFFLINE=1

export PRETRAINED_MODEL_PATH="thomas0829/pi05-pytorch-base"

RENAME_MAP='{"observation.images.front_camera":"observation.images.top","observation.images.left_camera":"observation.images.left","observation.images.right_camera":"observation.images.right"}'

accelerate launch \
    --num_processes=${n_gpus} \
    lerobot/scripts/accelerate_train.py \
    --compile=false \
    --policy.path="${PRETRAINED_MODEL_PATH}" \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"QUANTILES"}' \
    --rename_map="${RENAME_MAP}" \
    --dataset.repo_id="thomas0829/put_the_dolls_on_the_cloth" \
    --dataset.image_transforms.enable=true \
    --batch_size=1 \
    --steps=70000 \
    --save_freq=5000 \
    --strict=true \
    --num_workers=4 \
    --log_freq=10 \
    --gradient_accumulation_steps=2 \
    --policy.gradient_checkpointing=true \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.scheduler_decay_steps=30000 \
    --policy.optimizer_lr=2.5e-5 \
    --policy.repo_id="thomas0829/finetune_pi05" \
    --policy.push_to_hub=true \
    --policy.private=false \
    --dataset.use_annotated_tasks=false \
    --job_name=pi05_training \
    --wandb.enable=true \
    --wandb.project="yam-pi05-finetune" \
    --wandb.notes="Full fine-tuning of pi05 on put_the_dolls_on_the_cloth dataset"