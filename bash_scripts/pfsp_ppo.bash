#!/bin/bash

# --- Run: ppo_bus_20260409_110352 (5M steps, least_diverse pruning) ---
# python scripts/train.py \
#     --use_opponent_pool \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 50000 \
#     --total_timesteps 5000000 \
#     --ent_coef 0.03 \
#     --ent_coef_final 0.01 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.02 \
#     --lr 3e-4 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 20 \
#     --prune_strategy least_diverse \
#     --multi_policy \
#     --self_play_prob 0.3 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --sampling_method pfsp \
#     --load_pool_dir logs/ppo_bus_20260408_101719/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260408_101719/best_model/best_model.zip \
#     --start_fresh_directory

# --- Run: continuation from ppo_bus_20260409_110352 ---
# Changes vs. previous run:
#   - load_pool_dir/initial_checkpoint updated to current run (best by OpenSkill: ckpt_4950000)
#   - prune_strategy: least_diverse -> lowest_elo (evicts junk negative-mu checkpoints)
#   - self_play_prob: 0.3 -> 0.2 (more pool pressure at this skill level)
#   - ent_coef: 0.03->0.01 now 0.02->0.005 (policy is mature, reduce oscillation risk)
#   - pool_eval_games: 20 -> 2 (eval is greedy/deterministic, extra games are redundant)
#   - pool_eval_interval: 50000 -> 25000 (finer OpenSkill updates for better PFSP weights)
#   - prime_n_top: 20, prime_n_random: 5 (frontload strong checkpoints, small diversity buffer)
# python scripts/train.py \
#     --use_opponent_pool \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --total_timesteps 5000000 \
#     --ent_coef 0.02 \
#     --ent_coef_final 0.005 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.02 \
#     --lr 3e-4 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 2 \
#     --prune_strategy lowest_elo \
#     --multi_policy \
#     --self_play_prob 0.2 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --sampling_method pfsp \
#     --load_pool_dir logs/ppo_bus_20260409_110352/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260409_110352/opponent_pool/ckpt_4950000_20260410_030507.zip \
#     --start_fresh_directory \
#     --prime_n_top 20 \
#     --prime_n_random 5

# --- Run: continuation from ppo_bus_20260410_140002 ---
# Changes vs. previous run:
#   - load_pool_dir/initial_checkpoint updated to current run (best_pool_model: step 4976592, mu=4800)
#   - ent_coef: 0.02->0.005 now 0.01->0.002 (policy maturing, reduce oscillation further)
#   - pool_eval_games: 2 -> 4 (current pool has high-sigma entries; reduce rating noise)
#   - self_play_prob: 0.2 -> 0.15 (pool is now dense with strong opponents)
# python scripts/train.py \
#     --use_opponent_pool \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --total_timesteps 5000000 \
#     --ent_coef 0.01 \
#     --ent_coef_final 0.002 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.02 \
#     --lr 3e-4 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 4 \
#     --prune_strategy lowest_elo \
#     --multi_policy \
#     --self_play_prob 0.15 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --sampling_method pfsp \
#     --load_pool_dir logs/ppo_bus_20260410_140002/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260410_140002/best_pool_model/best_model.zip \
#     --start_fresh_directory \
#     --prime_n_top 20 \
#     --prime_n_random 5

# --- Run: continuation from ppo_bus_20260411_120820 ---
# Changes vs. previous run:
#   - initial_checkpoint: best_pool_model (step 4976592, mu=3904) -- peak-rated snapshot this run
#   - load_pool_dir: updated to current run
#   - lr: 3e-4 -> 5e-4 (KL was consistently ~0.007 vs target 0.02; only ~35% of update budget used)
#   - target_kl: 0.02 -> 0.025 (paired with higher lr; monitor for instability early on)
#   - n_epochs: 10 -> 15 (value_loss trended 0.06->0.31 and explained_variance dropped to 0.67;
#                         more passes per rollout gives value function more time to converge)
#   - self_play_prob: 0.15 -> 0.10 (pool is well-established; maximize pool pressure)
#   - openskill_recenter_interval: 1000000 -> 2000000 (halve recenter disruptions per 5M run;
#                                                       recenters were collapsing PFSP weights)
#   - pool_eval_games: 4 (unchanged; eval is deterministic)
# python scripts/train.py \
#     --use_opponent_pool \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --total_timesteps 5000000 \
#     --ent_coef 0.01 \
#     --ent_coef_final 0.002 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.025 \
#     --lr 5e-4 \
#     --n_epochs 15 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 4 \
#     --prune_strategy lowest_elo \
#     --multi_policy \
#     --self_play_prob 0.10 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --sampling_method pfsp \
#     --openskill_recenter_interval 2000000 \
#     --load_pool_dir logs/ppo_bus_20260411_120820/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260411_120820/best_pool_model/best_model.zip \
#     --start_fresh_directory \
#     --prime_n_top 20 \
#     --prime_n_random 5

# --- Run: continuation from ppo_bus_20260412_104509 ---
# Changes vs. previous run:
#   - initial_checkpoint: best_pool_model (step 3951264, mu=6223) -- peak-rated snapshot this run,
#                         captured just before the 2nd recenter at step 4M; do NOT use final model
#   - load_pool_dir: updated to current run
#   - openskill_recenter_interval: 2000000 -> 5000000 (one recenter per 5M run, late in training;
#                                                       2 recenters/run cost ~1-1.5M steps of clean
#                                                       PFSP signal as weights restabilize post-reset)
#   - lr: 5e-4 -> 7e-4 (KL consistently ~0.007-0.009 vs target 0.025; only ~30% of budget used;
#                        n_epochs=15 gives headroom to absorb the increase safely)
#   - target_kl: 0.025 -> 0.03 (paired with higher lr; still conservative)
#   - ent_coef: 0.01->0.002 now 0.005->0.001 (policy is mature; lower starting entropy,
#               floor at 0.001 to preserve minimal exploration)
#   - self_play_prob: 0.10 -> 0.05 (pool is well-established; maximize pool pressure)
#   - prime_n_top: 20 -> 25 (pool mu_spread narrowed to ~838 post-recenter; grab more strong
#                            opponents to seed better stratification from the start)
#   - n_epochs: 15 (unchanged; working well)
#   - pool_eval_games: 4 (unchanged)
# python scripts/train.py \
#     --use_opponent_pool \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --total_timesteps 5000000 \
#     --ent_coef 0.005 \
#     --ent_coef_final 0.001 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.03 \
#     --lr 7e-4 \
#     --n_epochs 15 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 4 \
#     --prune_strategy lowest_elo \
#     --multi_policy \
#     --self_play_prob 0.05 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --sampling_method pfsp \
#     --openskill_recenter_interval 5000000 \
#     --load_pool_dir logs/ppo_bus_20260413_114407/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260413_114407/best_pool_model/best_model.zip \
#     --start_fresh_directory \
#     --prime_n_top 25 \
#     --prime_n_random 5

# --- Run: fresh self-play priming with updated waste signals + slot-actionability ---
# New from scratch — no load_pool_dir or initial_checkpoint.
# Key changes vs. all prior runs:
#   - New reward signals active by default:
#       resolution_type1_waste_penalty -0.1 (Type 1 waste penalised at resolution time)
#       marker_opportunity_bonus: 0.02 -> 0.05
#       avoidable_waste_penalty:  -0.02 -> -0.1
#   - --use-slot-actionability: binary is_actionable feature per action-board slot
#       (obs dim +42; must start at a fresh checkpoint boundary, which this is)
#   - self_play_prob 1.0: pure self-play throughout — no external pool pressure during priming
#   - pool_eval_interval 0: pool eval disabled (no opponents to evaluate against yet)
#   - prune_strategy oldest: retain recent snapshots to track improvement trajectory
#   - lr 5e-4: history shows 3e-4 uses only ~30% of KL budget; 5e-4 is safer than jumping
#       straight to 7e-4 on an untrained policy
#   - ent_coef 0.05 -> 0.01: wide exploration early, narrow as policy matures over 5M steps
#   - openskill_recenter_interval 0: disable recentering — PFSP matchmaking weights are
#       irrelevant during pure self-play; avoids destabilising ratings mid-priming
#   - waste-debug-log: collect per-slot resolution detail to validate avoidable_rate metric
#       (Proposal 5); file is small and appended each eval run

###########--------------------------------##############
#### Self Play Priming (ppo_bus_20260415_114018) — COMPLETE ####
# Results: Type1 waste eliminated by step 550K, held at 0% for remaining 4.5M steps.
# KL ~27% of budget (same as all prior runs); explained_variance peaked 0.90, settled 0.68.
# Pool: 98 ckpts (steps 4.02M-5.005M), all mu=1500 sigma=433 (no games played, eval disabled).
# best_model saved ~step 4.38M (noisy 10-ep window peak); bus_model_final.zip used as seed.
# python scripts/train.py \
#     --use_opponent_pool \
#     --multi_policy \
#     --self_play_prob 1.0 \
#     --sampling_method pfsp \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 0 \
#     --prune_strategy oldest \
#     --total_timesteps 5000000 \
#     --ent_coef 0.05 \
#     --ent_coef_final 0.01 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.03 \
#     --lr 5e-4 \
#     --n_epochs 10 \
#     --n_eval_episodes 10 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 20 \
#     --pl_tau 30.0 \
#     --openskill_recenter_interval 0 \
#     --use-slot-actionability \
#     --waste-debug-log waste_debug_priming.jsonl

# --- Run: pool play from priming checkpoint ppo_bus_20260415_114018 ---
# Changes vs. self-play priming:
#   - initial_checkpoint: priming final model (bus_model_final.zip, 5M steps)
#   - load_pool_dir: priming pool (98 ckpts, all mu=1500 unrated, steps 4.02M-5.005M)
#   - prune_strategy: oldest -> lowest_elo (pool ratings now matter; evict weakest as new strong
#                     ckpts arrive; least_diverse ill-suited when all starting ratings are equal)
#   - pool_eval_interval: 0 -> 25000 (enable ratings; PFSP weights need signal)
#   - pool_eval_opponents: 10, pool_eval_games: 4 (established pattern from runs 3-5)
#   - self_play_prob: 1.0 -> 0.10 (pool exists now; can lower towards 0.05 if pool matures)
#   - lr: 5e-4 -> 7e-4 (KL was 27% of budget throughout priming, same as every prior run)
#   - n_epochs: 10 -> 15 (explained_variance settled at 0.68; value head needs more passes)
#   - ent_coef: 0.05->0.01 now 0.005->0.001 (policy mature after 5M priming steps)
#   - openskill_recenter_interval: 0 -> 5000000 (re-enable once at end; mid-run recenters
#                                                 destabilize PFSP weights for ~1-1.5M steps)
#   - skill_temperature: 20 -> 25 (pool ratings will spread; ease softmax toward 30 next run)
#   - prime_n_top 25, prime_n_random 5 (seed pool with best priming ckpts + diversity buffer)
#   - removed waste-debug-log (Type1 waste eliminated at ~550K steps; validation complete)
#   - --use-slot-actionability required (obs dim 27888 must match priming checkpoint)
# python scripts/train.py \
#     --use_opponent_pool \
#     --multi_policy \
#     --self_play_prob 0.10 \
#     --sampling_method pfsp \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 4 \
#     --prune_strategy lowest_elo \
#     --total_timesteps 5000000 \
#     --ent_coef 0.015 \
#     --ent_coef_final 0.005 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.03 \
#     --lr 7e-4 \
#     --n_epochs 15 \
#     --n_eval_episodes 10 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 25 \
#     --pl_tau 30.0 \
#     --openskill_recenter_interval 5000000 \
#     --load_pool_dir logs/ppo_bus_20260415_114018/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260415_114018/bus_model_final.zip \
#     --start_fresh_directory \
#     --prime_n_top 25 \
#     --prime_n_random 5 \
#     --use-slot-actionability

# --- Run: continuation from ppo_bus_20260415_204025 (ppo_bus_20260417_114306) — COMPLETE ---
# Results: policy mu=26975 (pool mean=26409, pool top=27080); win_rate=0.375 vs 12 pool opponents;
#          ep_len declined 149->138 (more decisive play); eval reward noisy 4-11.6 (expected in
#          self-play; note: 10 eval episodes are identical/seeded — effectively 1 game);
#          best model at step 4,951,584. KL remained ~27% of budget throughout (consistent
#          with all prior runs — shallow gradient regime, not a hyperparameter issue).
# Changes vs. previous run:
#   - initial_checkpoint: best_pool_model (step 4976592, mu=11313 pre-recenter)
#                         pool was recentered at 5M steps; carry clean post-recenter pool
#   - load_pool_dir: updated to current run (post-recenter, mu spread 645-2192, 4 unrated)
#   - ent_coef: 0.015->0.005 now 0.008->0.002 (policy ~10M total steps; further reduce
#               starting entropy; lower floor for more exploitation at this maturity)
#   - pool_eval_games: 4 -> 6 (policy is at pool ceiling; reduce variance in clustered
#                               top-tier ratings (mu 1900-2200) for sharper PFSP weights
#                               without excessive eval overhead)
#   - skill_temperature: 25 -> 30 (pool is fully stratified post-recenter; softer softmax
#                                   spreads training pressure across more of the pool)
#   - openskill_recenter_interval: 5000000 -> 0 (pool already freshly recentered; disable
#                                                 in-run recenter; prior runs showed
#                                                 recenters disrupt PFSP for ~1-1.5M steps)
#   - total_timesteps: 5000000 -> 10000000 (pool is well-calibrated from day 1; extend
#                                            to extract more learning from clean signal)
#   - prime_n_random: 5 -> 10 (seed more low-mu diversity from the 645-1500 range for
#                               better early-run PFSP stratification)
#   - lr, target_kl, n_epochs, self_play_prob: unchanged (KL ~27% budget regardless of
#     lr; policy gradients are in a shallow regime; all other params working well)
# python scripts/train.py \
#     --use_opponent_pool \
#     --multi_policy \
#     --self_play_prob 0.05 \
#     --sampling_method pfsp \
#     --pool_size 100 \
#     --pool_save_interval 10000 \
#     --pool_eval_interval 25000 \
#     --pool_eval_opponents 10 \
#     --pool_eval_games 6 \
#     --prune_strategy lowest_elo \
#     --total_timesteps 5000000 \
#     --ent_coef 0.005 \
#     --ent_coef_final 0.002 \
#     --n_envs 16 \
#     --n_steps 512 \
#     --batch_size 1024 \
#     --target_kl 0.03 \
#     --lr 7e-4 \
#     --n_epochs 15 \
#     --n_eval_episodes 10 \
#     --randomize_training_slot \
#     --disable_dist_validate \
#     --diag_log_interval 100000 \
#     --diag_log_samples 256 \
#     --diag_log_tolerance 5e-5 \
#     --skill_tracking openskill \
#     --skill_temperature 30 \
#     --pl_tau 30.0 \
#     --openskill_recenter_interval 0 \
#     --load_pool_dir logs/ppo_bus_20260416_132426/opponent_pool \
#     --initial_checkpoint logs/ppo_bus_20260416_132426/best_pool_model/best_model.zip \
#     --start_fresh_directory \
#     --prime_n_top 25 \
#     --prime_n_random 10 \
#     --use-slot-actionability

# --- Run: continuation from ppo_bus_20260417_114306 ---
# Changes vs. previous run:
#   - initial_checkpoint/load_pool_dir: updated to current run (best_pool_model step 4,951,584,
#                                       mu=26980; pool 100 ckpts steps 3.96M-5.0M)
#   - total_timesteps: 5000000 -> 6000000 (5M run showed policy still above pool average;
#                                           pool well-calibrated from day 1; extract more signal)
#   - pool_eval_games: 6 -> 10 (policy is at pool ceiling, mu spread ~1688 pts; 6 games/opponent
#                                is noisy when top-tier games are closely matched; 10 games gives
#                                more reliable PFSP weights without prohibitive eval overhead)
#   - n_steps: 512 -> 1024 (average ep_len=138 steps; 512-step rollouts cover ~3.7 episodes/env;
#                            1024-step rollouts cover ~7.4 — cleaner advantage estimates and
#                            better value function targets; explained_variance persistently ~0.68,
#                            longer rollouts reduce truncation bias in TD targets)
#   - batch_size: 1024 -> 2048 (maintain ~8 minibatches/epoch with the doubled rollout buffer;
#                                total_rollout = 16 envs * 1024 steps = 16384; 16384/2048 = 8)
#   - ent_coef_final: 0.002 -> 0.003 (raise entropy floor slightly; ~15M total training steps,
#                                      policy is in shallow gradient regime throughout all runs;
#                                      minimal floor increase may help escape local optima without
#                                      sacrificing the exploitation phase)
#   - openskill_recenter_interval: 0 -> 6000000 (mu has drifted to ~27K; re-enable a single
#                                                end-of-run recenter to standardize ratings for
#                                                the next run; fires at step 6M so no PFSP
#                                                disruption during training)
#   - lr, target_kl, n_epochs, self_play_prob, skill_temperature: all unchanged
#     (KL has been ~27% of budget at every lr; further lr changes are not the lever here)
python scripts/train.py \
    --use_opponent_pool \
    --multi_policy \
    --self_play_prob 0.02 \
    --sampling_method pfsp \
    --pool_size 100 \
    --pool_save_interval 10000 \
    --pool_eval_interval 25000 \
    --pool_eval_opponents 10 \
    --pool_eval_games 10 \
    --prune_strategy lowest_elo \
    --total_timesteps 2000000 \
    --ent_coef 0.001 \
    --ent_coef_final 0.0005 \
    --n_envs 16 \
    --n_steps 1024 \
    --batch_size 2048 \
    --target_kl 0.03 \
    --lr 7e-4 \
    --n_epochs 15 \
    --n_eval_episodes 10 \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --skill_tracking openskill \
    --skill_temperature 30 \
    --pl_tau 30.0 \
    --openskill_recenter_interval 2000000 \
    --load_pool_dir logs/ppo_bus_20260419_220322/opponent_pool \
    --initial_checkpoint logs/ppo_bus_20260419_220322/best_pool_model/best_model.zip \
    --start_fresh_directory \
    --prime_n_top 30 \
    --prime_n_random 10 \
    --use-slot-actionability
