#!/bin/bash

python scripts/train.py \
    --use-opponent-pool \
    --multi-policy \
    --pool-size 20 \
    --sampling-method pfsp \
    --self-play-prob 0.2 \
    --pool-eval-interval 100000 \
    --total-timesteps 5000000 \
    --ent_coef 0.1 \
    --ent-coef-final 0.01 \
    --n-envs 16