#!/bin/bash
# set -euo pipefail

# Self-play bootstrap run to learn core strategy before heavier pool pressure.
python scripts/train.py \
    --use_opponent_pool \
    --multi_policy \
    --self_play_prob 1.0 \
    --sampling_method pfsp \
    --pool_size 50 \
    --pool_save_interval 10000 \
    --pool_eval_interval 0 \
    --prune_strategy oldest \
    --total_timesteps 2000000 \
    --ent_coef 0.05 \
    --ent_coef_final 0.02 \
    --n_envs 16 \
    --n_steps 512 \
    --batch_size 1024 \
    --target_kl 0.02 \
    --lr 3e-4 \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --skill_tracking openskill \
    --skill_temperature 20 \
    --pl_tau 30.0

# Optional continuation seed inputs once bootstrap completes:
# --load_pool_dir logs/ppo_bus_YYYYMMDD_HHMMSS/opponent_pool \
# --initial_checkpoint logs/ppo_bus_YYYYMMDD_HHMMSS/checkpoints/bus_model_1000000_steps.zip \
# --start_fresh_directory
