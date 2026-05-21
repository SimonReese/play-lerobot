#!/bin/bash

# Lauch lerobot train for pi05_exp
lerobot-train \
    --job_name pi05_exp_evo \
    --steps 50000 \
    --log_freq 100 \
    --batch_size 8 \
    --save_checkpoint true \
    --dataset.repo_id SimonReese/lerobot-20-ep-v3 \
    --dataset.root ./datasets/lerobot-20-ep-v3 \
    --policy.type pi05_exp \
    --policy.pretrained_path lerobot/pi05_base \
    --policy.train_expert_only true \
    --policy.device cuda \
    --policy.push_to_hub false \
    --policy.gradient_checkpointing true \
    --policy.dtype bfloat16 \
    --wandb.enable true \
    --wandb.entity simon-reese-personal \
    --wandb.project pi05_exp_evo \
    --wandb.run_id pi05_exp_evo \
    --output_dir outputs/train/pi05_exp_evo