#!/bin/bash

export LD_LIBRARY_PATH=/nfs/guille/tgd/wonglab/dinkinst/conda_install/envs/Bus/lib:${LD_LIBRARY_PATH}

# ============================================================================
# AlphaZero Bootstrap Pipeline — PPO warm-start → full MCTS
# ============================================================================
#
# MOTIVATION:
#   Cold-start AlphaZero on Bus stalls: a random policy gives MCTS no traction,
#   all games end near-uniform, and the value head sees no signal. Behavioral
#   cloning (BC) from a trained MaskablePPO model warm-starts the AlphaZeroNetwork
#   with a delivery-aware prior so MCTS immediately finds scoring lines within
#   its simulation budget.
#
# USAGE PATTERN:
#   Run phases sequentially. Once a phase completes successfully, comment it out
#   and uncomment the next phase. This avoids accidentally re-running expensive
#   steps and makes it easy to see where you are in the pipeline.
#
# PREREQUISITE:
#   A trained MaskablePPO model (trained with --use-slot-actionability).
#   Update RUNNAME below to match your PPO run directory.
#
# FULL SEQUENCE:
#   Phase 0  Collect ~1000 BC game records from PPO (greedy)
#   Phase 1  Supervised pre-train a fresh AlphaZeroNetwork on those records
#   Phase 2  AlphaZero Run 1: BC-warmed, low sims (50), reward shaping ON
#            Goal: eval/avg_rank > 0.52 within 100 iters
#   Phase 3  AlphaZero Run 2: ramp-up (200 sims)
#            Goal: eval/avg_rank trending past 0.55
#   Phase 4  AlphaZero Run 3: full strength (400 sims)
#            Goal: plateau detection, stop when rank stalls >50 iters
# ============================================================================

# Edit this to match your PPO run directory name.
RUNNAME="20260502_112313"

# ============================================================================
# PHASE 0: Collect BC game records from pretrained PPO
# ============================================================================
#
# Runs N games with the PPO policy (greedy) and saves (obs, head_id,
# action_probs, value_target) for every step. The output pickle is used by
# Phase 1.
#
# Suggested N: 1000 games (~100k–150k moves). This takes 15–40 minutes on
# CPU depending on game length. Increase to 2000 games for a richer dataset
# if the BC policy loss doesn't converge (final pol loss should be < 1.5).
#
# When complete: comment out this block and uncomment Phase 1.
#
# python scripts/collect_bc_data.py \
#     --ppo_checkpoint "logs/ppo_bus_${RUNNAME}/best_pool_model/best_model.zip" \
#     --n_games 2000 \
#     --output logs/bc_data/bc_data.pkl \
#     --use_slot_actionability \
#     --num_players 4 \
#     --no-greedy \
#     --device auto


# ============================================================================
# PHASE 1: BC pre-training of AlphaZeroNetwork
# ============================================================================
#
# Trains a fresh AlphaZeroNetwork for 10 epochs on the BC records collected
# above. Uses per-phase heads to match the planned AlphaZero architecture.
#
# Expected final losses (per-phase mode, 1000 games, 10 epochs):
#   policy: 1.0–2.0 (lower means closer to PPO distribution)
#   value:  0.08–0.15 (lower means value head tracks game outcomes)
#
# If policy loss stalls > 2.5 after epoch 5, increase --n_games in Phase 0
# or add --epochs 15 here.
#
# When complete: comment out this block and uncomment Phase 2.
#
# python scripts/bc_pretrain.py \
#     --bc_data logs/bc_data/bc_data.pkl \
#     --output logs/bc_warmstart/network.pt \
#     --epochs 30 \
#     --batch_size 512 \
#     --lr 1e-3 \
#     --trunk_layers 512 512 256 \
#     --use_per_phase_heads \
#     --use_slot_actionability \
#     --device auto


# ============================================================================
# PHASE 2: AlphaZero Run 1 — BC-warmed, low simulations
# ============================================================================
#
# First AlphaZero run starting from the BC checkpoint. Low sim count (50) and
# small game budget (25/iter) for fast iteration. Reward shaping ON to
# supplement sparse terminal signal during the warm-start transition.
#
# Lower LR (5e-4 vs cold-start 1e-3): the BC-warmed network already has a
# useful prior; smaller updates prevent early low-quality MCTS samples from
# overwriting BC knowledge.
#
# New in this revision:
#   --dirichlet_epsilon 0.15   Reduced from 0.25; BC prior carries real info,
#                              25% noise cancels too much of it at this stage.
#   --eval_games 40            SE ~0.08 instead of 0.16; noise floor was
#                              masking all genuine improvement signal.
#   --use_reward_shaping       Blends delivery step-rewards into z (alpha=0.15)
#                              so the value head sees delivery signal early.
#   --bc_checkpoint …          Frozen BC network used for:
#                                (a) KL regularization: anchors current policy
#                                    toward BC prior (weight decays 0.1 → 0
#                                    over 100 iters).
#                                (b) Mid-game curriculum: plays round 1 with BC
#                                    so MCTS always starts in delivery-imminent
#                                    positions (--bc_rounds 1).
#   --rollout_to_round_end     Each leaf evaluation rolls forward with the
#                              greedy policy until the next CHOOSING_ACTIONS
#                              phase, giving MCTS a full resolution-cycle
#                              horizon (~20-35 extra steps). This is the primary
#                              fix for the shallow-tree problem.
#   --rollout_reward_weight 0.3 Blends rollout delivery rank (30%) with
#                              network value (70%) at each leaf.
#
# Goal: eval/avg_rank > 0.52 (above random baseline of 0.50) by iteration 50.
# If it stays flat through iteration 60, check TensorBoard:
#   - policy_loss should be trending down from ~2.0 toward ~1.5
#   - value_loss should be below 0.12 by iter 30
#   - eval/avg_rank should show an upward trend, not flat 0.50
#   - Deliveries per game (self-play log) should be > 0 within 20 iters
#
# When Run 1 stabilizes: comment out this block and uncomment Phase 3.
#
python scripts/train_mcts.py \
    --iterations 75 \
    --games_per_iter 10 \
    --n_simulations 50 \
    --use_slot_actionability \
    --use_per_phase_heads \
    --n_workers 1 \
    --trunk_layers 512 512 256 \
    --train_steps 500 \
    --batch_size 512 \
    --replay_buffer_size 50000 \
    --min_buffer_size 2500 \
    --lr 7e-4 \
    --max_grad_norm 2.0 \
    --c_puct 1.5 \
    --dirichlet_alpha 0.3 \
    --dirichlet_epsilon 0.2 \
    --temperature_threshold 30 \
    --eval_games 60 \
    --eval_rank_threshold 0.52 \
    --eval_every 5 \
    --save_every 5 \
    --checkpoint_dir logs/alphazero_bc_run1 \
    --initial_checkpoint logs/bc_warmstart/network.pt \
    --use_reward_shaping \
    --bc_checkpoint logs/bc_warmstart/network.pt \
    --kl_weight 0.02 \
    --kl_total_iters 50 \
    --bc_rounds 1 \
    --rollout_to_round_end \
    --rollout_reward_weight 0.3 \
    --device auto \
    --tensorboard


# ============================================================================
# PHASE 3: AlphaZero Run 2 — Ramp-up
# ============================================================================
#
# Resume from Run 1 best checkpoint. Scale up simulations (200) and
# games/iter (50). Reward shaping kept ON — the network is still learning
# delivery value.
#
# Changes vs Run 1:
#   n_simulations:      50  → 200  (4x better search quality; ~4x slower/iter)
#   games_per_iter:     25  → 50   (richer data per iteration)
#   train_steps:       500  → 1000 (more gradient steps for larger batches)
#   replay_buffer:   50000  → 150000
#   min_buffer_size:  2500  → 7500
#   lr:              5e-4   → 3e-4  (continue reducing to preserve learned policy)
#   dirichlet_epsilon: 0.15 → 0.25  (ramp noise back up now that BC prior is less critical)
#   eval_rank_threshold: 0.52 → 0.54
#   rollout_steps:   round-end → 15  (taper rollout depth as value fn matures)
#   rollout_reward_weight: 0.3 → 0.1
#   bc_rounds: 1 → 2         (if deliveries still sparse, go deeper into curriculum)
#   kl_weight: 0.0           (KL anchor no longer needed; policy has stabilized)
#
# Monitor train/v_std in TensorBoard: once it consistently exceeds ~0.35 the
# value head is differentiating positions well enough to reduce rollouts.
#
# Update --initial_checkpoint to the actual Run 1 incumbent path.
# Goal: eval/avg_rank > 0.55 by iteration 100.
#
# When Run 2 stabilizes: comment out this block and uncomment Phase 4.
#
# python scripts/train_mcts.py \
#     --iterations 200 \
#     --games_per_iter 50 \
#     --n_simulations 200 \
#     --use_slot_actionability \
#     --use_per_phase_heads \
#     --n_workers 1 \
#     --trunk_layers 512 512 256 \
#     --train_steps 1000 \
#     --batch_size 512 \
#     --replay_buffer_size 150000 \
#     --min_buffer_size 7500 \
#     --lr 3e-4 \
#     --max_grad_norm 1.0 \
#     --c_puct 1.5 \
#     --dirichlet_alpha 0.3 \
#     --dirichlet_epsilon 0.25 \
#     --temperature_threshold 30 \
#     --eval_games 40 \
#     --eval_rank_threshold 0.54 \
#     --eval_every 5 \
#     --save_every 5 \
#     --checkpoint_dir logs/alphazero_bc_run2 \
#     --initial_checkpoint logs/alphazero_bc_run1/incumbent.pt \
#     --use_reward_shaping \
#     --bc_checkpoint logs/bc_warmstart/network.pt \
#     --bc_rounds 2 \
#     --rollout_steps 15 \
#     --rollout_reward_weight 0.1 \
#     --device auto \
#     --tensorboard


# ============================================================================
# PHASE 4: AlphaZero Run 3 — Full strength
# ============================================================================
#
# Full AlphaZero regime. Resume from Run 2 best checkpoint.
# Run until eval/avg_rank plateaus for >50 consecutive iterations.
# No rollouts, no KL anchoring, no curriculum — pure self-play with a
# mature value function.
#
# Changes vs Run 2:
#   n_simulations:    200  → 400  (full AlphaZero quality; ~2x slower/iter)
#   games_per_iter:    50  → 100  (fill 200k buffer faster)
#   replay_buffer: 150000  → 200000
#   min_buffer_size:  7500 → 15000 (≈1 full iter of data before training)
#   lr:              3e-4  → 2e-4
#   eval_rank_threshold: 0.54 → 0.56
#   rollout_steps:  15 → 0  (value head should stand on its own by now)
#   use_reward_shaping: consider disabling if value head is well-calibrated
#
# GPU strongly recommended at 400 sims (CPU would take ~3-7 days per 100 iters).
#
# Update --initial_checkpoint to the actual Run 2 incumbent path.
#
# python scripts/train_mcts.py \
#     --iterations 300 \
#     --games_per_iter 100 \
#     --n_simulations 400 \
#     --use_slot_actionability \
#     --use_per_phase_heads \
#     --n_workers 1 \
#     --trunk_layers 512 512 256 \
#     --train_steps 1000 \
#     --batch_size 512 \
#     --replay_buffer_size 200000 \
#     --min_buffer_size 15000 \
#     --lr 2e-4 \
#     --max_grad_norm 1.0 \
#     --c_puct 1.5 \
#     --dirichlet_alpha 0.3 \
#     --dirichlet_epsilon 0.25 \
#     --temperature_threshold 30 \
#     --eval_games 40 \
#     --eval_rank_threshold 0.56 \
#     --eval_every 5 \
#     --save_every 5 \
#     --checkpoint_dir logs/alphazero_bc_run3 \
#     --initial_checkpoint logs/alphazero_bc_run2/incumbent.pt \
#     --device auto \
#     --tensorboard
