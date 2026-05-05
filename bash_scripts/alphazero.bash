#!/bin/bash

# ============================================================================
# AlphaZero training for Bus — principled run sequence
# ============================================================================
#
# STRUCTURE: Three phases, each resuming from the previous.
#   Run 1 — Bootstrap (50 sims, 25 games/iter, 100 iters)
#     Purpose: establish a policy that plays legal moves and learns basic game
#     structure. The value head starts near 0.25 MSE (uniform output over [0,1])
#     and should fall below 0.15 by the end. Policy loss starts at
#     -log(1/num_valid_actions) ≈ 3-5 and should fall steadily.
#     Estimated time: ~2-4 h on GPU, 8-16 h on CPU.
#
#   Run 2 — Ramp-up (200 sims, 50 games/iter, 200 iters)
#     Purpose: scale up search quality now that the network has learned to
#     distinguish good from bad moves. The larger replay buffer (200 k samples)
#     and better MCTS estimates should drive consistent policy improvement.
#     eval/avg_rank should climb above 0.55 (random baseline = 0.50) by ~iter 50.
#     Estimated time: ~12-24 h on GPU.
#
#   Run 3 — Full strength (400 sims, 100 games/iter, 300+ iters)
#     Purpose: full AlphaZero regime. Network should continue improving as long
#     as eval/avg_rank trends upward. Stop when it plateaus for >50 iterations.
#     GPU strongly recommended (CPU would take 3-7 days per 100 iters at 400 sims).
#
# TENSORBOARD:
#   Enable with --tensorboard. Logs are written to {checkpoint_dir}/tb/.
#   Launch viewer:  tensorboard --logdir logs/alphazero/tb
#   Logged scalars:
#     train/policy_loss  — cross-entropy vs MCTS visit distribution (lower = better)
#     train/value_loss   — MSE vs terminal rank vector z (lower = better)
#     train/total_loss   — weighted sum
#     eval/avg_rank      — new checkpoint's avg normalized rank vs previous (0–1,
#                          higher = better; 0.5 = random in 4-player game)
#   TensorBoard only logs training steps; the eval scalar fires every
#   --eval_every iterations when a previous checkpoint exists to compare against.
#
# RESUMING A RUN:
#   Pass --initial_checkpoint to the .pt file saved by the previous run.
#   Example checkpoint path: logs/alphazero/checkpoint_0100.pt
#   The trainer will also look for a matching _optim.pt sidecar to restore the
#   Adam optimizer state. The replay buffer is NOT persisted across runs — it
#   refills from self-play in the first few iterations.
#
# HOW MANY RUNS?
#   Typically 2-3 runs totalling 500-700 iterations (≈ 50-100 k games).
#   Watch eval/avg_rank: if it stays flat for 50 consecutive iterations despite
#   the buffer being full, either the network capacity is saturated (try larger
#   --trunk_layers) or search quality is the bottleneck (increase --n_simulations).
# ============================================================================


# ── Run 1: Bootstrap ─────────────────────────────────────────────────────────
# First run from scratch. Low sim count (50) and small game budget (25/iter)
# so you can verify the pipeline end-to-end quickly and get an initial policy.
# min_buffer_size set to 2500 (≈ 1 full iteration of data); training begins
# immediately after the first self-play batch.
# save_every=5 / eval_every=5 means you get a checkpoint every ~5 iters and
# a rank comparison every 5 iters once a previous checkpoint exists (iter 6+).
# python scripts/train_mcts.py \
#     --iterations 100 \
#     --games_per_iter 25 \
#     --n_simulations 50 \
#     --n_workers 1 \
#     --trunk_layers 512 512 256 \
#     --train_steps 500 \
#     --batch_size 512 \
#     --replay_buffer_size 50000 \
#     --min_buffer_size 2500 \
#     --lr 1e-3 \
#     --max_grad_norm 1.0 \
#     --c_puct 1.5 \
#     --dirichlet_alpha 0.3 \
#     --dirichlet_epsilon 0.25 \
#     --temperature_threshold 30 \
#     --eval_games 10 \
#     --eval_rank_threshold 0.52 \
#     --eval_every 5 \
#     --save_every 5 \
#     --checkpoint_dir logs/alphazero_run1 \
#     --device auto \
#     --tensorboard


# ── Run 2: Ramp-up ───────────────────────────────────────────────────────────
# Resume from the best Run 1 checkpoint. Scale up simulations (200) and
# games-per-iteration (50) to produce higher-quality training signal.
# Changes vs. Run 1:
#   - initial_checkpoint: last/best .pt from Run 1 (update path below)
#   - n_simulations: 50 -> 200 (better search quality; runtime ~4x longer/iter)
#   - games_per_iter: 25 -> 50 (larger, richer data batches each iteration)
#   - train_steps: 500 -> 1000 (more gradient steps per larger game batch)
#   - replay_buffer_size: 50000 -> 150000 (hold more history for stability)
#   - min_buffer_size: 2500 -> 7500 (wait for ≈1.5 iters of data before training)
#   - lr: 1e-3 -> 5e-4 (network is no longer random; reduce step size)
#   - eval_games: 10 -> 20 (more games = lower variance rank estimate)
#   - eval_rank_threshold: 0.52 -> 0.55 (tighter bar for promotion)
#   - checkpoint_dir: new directory to keep runs isolated
# python scripts/train_mcts.py \
#     --iterations 200 \
#     --games_per_iter 50 \
#     --n_simulations 200 \
#     --n_workers 1 \
#     --trunk_layers 512 512 256 \
#     --train_steps 1000 \
#     --batch_size 512 \
#     --replay_buffer_size 150000 \
#     --min_buffer_size 7500 \
#     --lr 5e-4 \
#     --max_grad_norm 1.0 \
#     --c_puct 1.5 \
#     --dirichlet_alpha 0.3 \
#     --dirichlet_epsilon 0.25 \
#     --temperature_threshold 30 \
#     --eval_games 20 \
#     --eval_rank_threshold 0.55 \
#     --eval_every 5 \
#     --save_every 5 \
#     --checkpoint_dir logs/alphazero_run2 \
#     --initial_checkpoint logs/alphazero_run1/checkpoint_0100.pt \
#     --device auto \
#     --tensorboard


# ── Run 3: Full strength ──────────────────────────────────────────────────────
# Full AlphaZero regime. Resume from the best Run 2 checkpoint.
# Changes vs. Run 2:
#   - initial_checkpoint: best .pt from Run 2 (update path below)
#   - n_simulations: 200 -> 400 (full AlphaZero quality; runtime ~2x longer/iter)
#   - games_per_iter: 50 -> 100 (fill 200k replay buffer faster; also
#                                 amortises the fixed MCTS overhead per iteration)
#   - train_steps: 1000 -> 1000 (unchanged; buffer is large enough)
#   - replay_buffer_size: 150000 -> 200000 (full capacity)
#   - min_buffer_size: 15000 (≈ 1 iter of data at 100 games × ~150 moves/game)
#   - lr: 5e-4 -> 2e-4 (network is maturing; reduce LR to preserve learned policy)
#   - eval_rank_threshold: 0.55 -> 0.58 (raise bar as policy improves)
#   - iterations: 300+ (run until eval/avg_rank plateaus for >50 consecutive iters)
python scripts/train_mcts.py \
    --iterations 300 \
    --games_per_iter 100 \
    --n_simulations 400 \
    --n_workers 1 \
    --trunk_layers 512 512 256 \
    --train_steps 1000 \
    --batch_size 512 \
    --replay_buffer_size 200000 \
    --min_buffer_size 15000 \
    --lr 2e-4 \
    --max_grad_norm 1.0 \
    --c_puct 1.5 \
    --dirichlet_alpha 0.3 \
    --dirichlet_epsilon 0.25 \
    --temperature_threshold 30 \
    --eval_games 20 \
    --eval_rank_threshold 0.58 \
    --eval_every 5 \
    --save_every 5 \
    --checkpoint_dir logs/alphazero_run3 \
    --initial_checkpoint logs/alphazero_run2/checkpoint_0200.pt \
    --device auto \
    --tensorboard
