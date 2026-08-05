#!/bin/bash
# =============================================================================
# pfsp_ppo_v2.bash — Bus PPO training plan with explicit reward config
# =============================================================================
#
# This file is structured as:
#   Phase 1 (ACTIVE): fresh self-play priming under the new reward regime.
#   Phase 2 (COMMENTED): PFSP pool play started from the completed Phase 1 run.
#
# --- Why start a new Phase 1 instead of loading the previous checkpoint? ---
#
#   Four changes here collectively make a clean Phase 1 necessary:
#
#   1. use_score_based_terminal: the value function must now predict absolute
#      delivery points rather than rank-differential. A checkpoint trained on
#      rank-differential has a value head that returns fundamentally different
#      numbers; loading it and switching terminal modes yields corrupted value
#      targets for the first several thousand gradient steps.
#
#   2. vf_coef 0.5 → 1.0: doubles the gradient contribution of the value loss.
#      Combined with the target change above, the optimizer trajectory from a
#      loaded checkpoint is unlikely to be useful.
#
#   3. gae_lambda 0.98 → 1.0: switches from TD(λ) bootstrapping to pure Monte
#      Carlo returns. Advantages computed from a mismatched value baseline will
#      have high bias until the value function re-calibrates; starting fresh is
#      cheaper than the recalibration period from a loaded checkpoint.
#
#   4. delivery_reward 1.75 → 2.5: changes the scale of the policy gradient
#      signal at every step. This alone does not require a fresh start but
#      compounds the mismatch from (1-3).
#
#   If compute is severely constrained, loading the existing best_pool_model is
#   still possible; expect ~500K steps of confused value estimates before the
#   value head recalibrates.
#
# --- Reward regime design ---
#
#   Sparse reward regime: only passenger deliveries, a small station-connection
#   bonus, and score-based terminal reward are active. All other shaping
#   (stolen/exclusive delivery bonuses, time-stone penalty, vrroomm shaping,
#   marker/resolve bonuses) has been removed to reduce noise and let credit
#   assignment flow through actual delivery outcomes and the terminal signal.
#
#   • use_score_based_terminal: terminal = (score - time_stones) * 1.0
#       Completely decouples the terminal reward from opponent scores.
#       The policy is rewarded for absolute delivery count, not margin of
#       victory. This removes the PFSP "win without scoring" pathology where
#       the policy beats weak pool opponents by narrow margins without
#       maximising deliveries.
#       won_game_bonus / draw_bonus / second_place_bonus are ignored.
#
#   • delivery_reward 2.5:
#       Keeps delivery rewards as the dominant per-step signal. With ~6
#       deliveries per game, delivery shaped reward ≈ 15; terminal reward for
#       a 7-point game ≈ 7. Terminal is ~32% of total shaped reward — enough
#       weight to matter but not so much that it swamps step-level credit.
#
#   • station_connection_reward 0.05:
#       Small first-time bonus for connecting to a train station. Provides
#       a lightweight directional signal for network building without
#       dominating the delivery-centric credit structure.
#
#   • gae_lambda 1.0 (Monte Carlo returns):
#       Removes all bootstrapping bias from advantage estimates. The value
#       function no longer contaminates advantages with its own prediction
#       error; advantages are computed from actual sampled returns to episode
#       end. Higher variance, but unbiased credit assignment for terminal-heavy
#       reward structures.
#
#   • vf_coef 1.0:
#       Doubles the value function gradient. With sparse terminal reward, the
#       value function is the bottleneck for credit assignment quality; giving
#       it the same loss weight as the policy loss is appropriate.
#
# --- What to watch ---
#   • eval/mean_game_score (TensorBoard): actual pts = player.score - stones.
#       This is the ground truth metric. Should trend toward 7-10 range.
#   • rollout/mean_game_score: same metric during training rollouts.
#   • train/value_loss: should decline steadily from a high initial value
#       (value head is starting from scratch against new targets). If it
#       plateaus above ~0.5, check that terminal_score_scale is not too large.
#   • train/approx_kl: should stay active; a steady-state KL near 0 indicates
#       the policy stopped learning.
#   • rollout/ep_len_true: stable around 110-140 steps is healthy.
#
# =============================================================================


# =============================================================================
# Phase 1: Fresh self-play priming under the new reward regime
# =============================================================================
#
# Goals:
#   - Build a pool of checkpoints that were optimised for absolute delivery
#     count from the start, not rank-differential.
#   - Give the value head clean data to learn the new terminal target before
#     introducing adversarial PFSP pressure.
#   - 100% self-play so the policy adapts without inheriting the old pool's
#     relative-scoring biases.
#
# Hyperparameter notes:
#   - gae_lambda 1.0: MC returns; no bootstrapping bias. At episode length
#     ~125 steps, MC variance is manageable; any bias from bootstrapping an
#     incorrect value function outweighs the variance reduction benefit.
#   - vf_coef 1.0: equal loss weight for policy and value. The value function
#     needs a stronger gradient to learn the new terminal target quickly.
#   - terminal_score_scale 1.0: 1 raw delivery point = 1.0 terminal reward
#     unit. With delivery_reward=2.5 per step, terminal is ~30% of total
#     shaped reward over an average game.
#   - n_envs 16, n_steps 4096: 65536-step rollout covers ~520 complete games
#     per update, giving the MC return estimator good coverage.
#   - ent_coef 0.02 → 0.006: exploration decay to match previous regime.
#   - pool_eval_interval 0: skip pool evaluation in Phase 1 since the pool
#     is sparse early and head-to-head noise would dominate the signal.
#

python scripts/train.py \
    --use_opponent_pool \
    --multi_policy \
    --self_play_prob 1.0 \
    --sampling_method pfsp \
    --pool_size 120 \
    --save_freq 262144 \
    --eval_freq 262144 \
    --pool_save_interval 131072 \
    --pool_eval_interval 0 \
    --prune_strategy oldest \
    --total_timesteps 8000000 \
    --ent_coef 0.02 \
    --ent_coef_final 0.006 \
    --n_envs 16 \
    --n_steps 4096 \
    --batch_size 2048 \
    --target_kl 0.015 \
    --lr 5e-4 \
    --n_epochs 10 \
    --gamma 0.995 \
    --gae_lambda 1.0 \
    --vf_coef 1.0 \
    --n_eval_episodes 24 \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --skill_tracking openskill \
    --skill_temperature 30 \
    --pl_tau 30.0 \
    --openskill_recenter_interval 0 \
    --use_score_based_terminal \
    --terminal_score_scale 1.0 \
    --delivery_reward 2.5 \
    --station_connection_reward 0.05 \
    --use-slot-actionability \
    --use-delivery-features


# =============================================================================
# Phase 2: PFSP pool play after Phase 1 completes
# =============================================================================
#
# Start this after Phase 1 finishes. Update PHASE1_RUN below before running.
#
# Goals:
#   - Introduce adversarial diversity via pool opponents trained under the
#     same score-based terminal regime.
#   - 20% self-play keeps the policy from fully regressing to pool-average
#     behaviour while 80% pool pressure breaks late self-play equilibria.
#   - delivery_reward bumped slightly to 2.75 so delivery shaping remains
#     the dominant signal as pool opponents improve.
#
# Changes vs Phase 1:
#   - self_play_prob 1.0 → 0.20 (introduce PFSP pool pressure)
#   - pool_eval_interval enabled (262144) for OpenSkill rating updates
#   - pool_eval_opponents 8: enough opponents for a stable mu estimate
#   - skill_temperature 150: softer pool sampling to avoid over-focusing
#     on a tiny elite subset
#   - delivery_reward 2.5 → 2.75
#   - ent_coef 0.012 → 0.003: tighter exploration as policy matures
#   - n_envs 14: one env freed for pool eval subprocess overhead
#   - prune_strategy least_diverse: preserve pool coverage
#
# To run:
#   1. Set PHASE1_RUN to the Phase 1 log directory name (e.g., ppo_bus_20260525_120000)
#   2. Uncomment the python command below and comment out Phase 1 above.
#
# PHASE1_RUN="ppo_bus_20260525_154410"
#
# python scripts/train.py \
#     --use_opponent_pool \
#     --multi_policy \
#     --self_play_prob 0.20 \
#     --sampling_method pfsp \
#     --pool_size 100 \
#     --save_freq 262144 \
#     --eval_freq 262144 \
#     --pool_save_interval 131072 \
#     --pool_eval_interval 262144 \
#     --pool_eval_opponents 8 \
#     --pool_eval_games 6 \
#     --prune_strategy least_diverse \
#     --total_timesteps 8000000 \
#     --ent_coef 0.012 \
#     --ent_coef_final 0.003 \
#     --n_envs 14 \
#     --n_steps 4096 \
#     --batch_size 2048 \
#     --target_kl 0.015 \
#     --lr 5e-4 \
#     --n_epochs 10 \
#     --gamma 0.995 \
#     --gae_lambda 1.0 \
#     --vf_coef 1.0 \
#     --n_eval_episodes 24 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 150 \
#     --pl_tau 30.0 \
#     --openskill_recenter_interval 0 \
#     --start_fresh_directory \
#     --prime_n_top 30 \
#     --prime_n_random 15 \
#     --load_pool_dir "logs/${PHASE1_RUN}/opponent_pool" \
#     --initial_checkpoint "logs/${PHASE1_RUN}/best_model/best_model.zip" \
#     --use_score_based_terminal \
#     --terminal_score_scale 1.0 \
#     --delivery_reward 2.75 \
#     --station_connection_reward 0.05 \
#     --use-slot-actionability \
#     --use-delivery-features
