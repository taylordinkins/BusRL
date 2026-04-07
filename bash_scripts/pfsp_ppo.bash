#!/bin/bash

python scripts/train.py \
    --use_opponent_pool \
    --pool_size 50 \
    --pool_save_interval 10000 \
    --pool_eval_interval 50000 \
    --total_timesteps 5000000 \
    --ent_coef 0.1 \
    --ent_coef_final 0.02 \
    --n_envs 16 \
    --n_steps 512 \
    --batch_size 1024 \
    --target_kl 0.02 \
    --lr 3e-4 \
    --pool_eval_opponents 10 \
    --pool_eval_games 10 \
    --prune_strategy lowest_elo \
    --multi_policy \
    --self_play_prob 0.2 \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --skill_tracking openskill \
    --skill_temperature 200 \
    --pl_tau 30.0 \
    --sampling_method elo_weighted
    # --load_pool_dir logs/ppo_bus_20260126_194851/opponent_pool \
    # --initial_checkpoint logs/ppo_bus_20260126_194851/opponent_pool/ckpt_2218544_20260203_003647 \
    # --start_fresh_directory \
    
