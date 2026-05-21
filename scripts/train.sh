#!/bin/bash

# Lauch lerobot train for pi05_exp
lerobot-train \
    --job_name pi05_exp_titan \
    --steps 1000 \
    --log_freq 100 \
    --batch_size 4 \
    --save_checkpoint false \
    --dataset.repo_id SimonReese/lerobot-20-ep-v3 \
    --dataset.root ./datasets/lerobot-20-ep-v3 \
    --policy.type pi05_exp \
    --policy.pretrained_path lerobot/pi05_base \
    --policy.train_expert_only true \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.gradient_checkpointing true \
    --policy.dtype bfloat16
    --wandb.enable true \
    --wandb.entity simon-reese-personal \
    --wandb.project pi05_exp \
    --output_dir outputs/train/pi05_exp_titan