#!/bin/bash

python scripts/train.py \
    --use-opponent-pool \
    --pool-size 40 \
    --pool-eval-interval 100000 \
    --total-timesteps 10000000 \
    --ent_coef 0.1 \
    --ent-coef-final 0.001 \
    --n-envs 8 \
    --n_steps 8192 \
    --batch_size 1024 \
    --target_kl 0.01 \
    --lr 1e-4 \
    --pool-eval-opponents 8 \
    --prune-strategy lowest_elo
