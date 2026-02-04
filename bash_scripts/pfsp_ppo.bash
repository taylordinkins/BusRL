#!/bin/bash

python scripts/train.py \
    --use_opponent_pool \
    --pool_size 20 \
    --pool_eval_interval 10000 \
    --total_timesteps 5000000 \
    --ent_coef 0.1 \
    --ent_coef_final 0.01 \
    --n_envs 16 \
    --n_steps 512 \
    --batch_size 1024 \
    --target_kl 0.02 \
    --lr 1e-4 \
    --pool_eval_opponents 20 \
    --pool_eval_games 6 \
    --prune_strategy lowest_elo \
    --multi_policy \
    --self_play_prob 0.2 \
    --sampling_method pfsp \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --load_pool_dir logs/ppo_bus_20260126_194851/opponent_pool \
    --initial_checkpoint logs/ppo_bus_20260126_194851/opponent_pool/ckpt_2218544_20260203_003647 \
    --start_fresh_directory
