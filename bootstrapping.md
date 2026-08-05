# PPO Bootstrap for AlphaZero MCTS

## Background

Cold-start AlphaZero MCTS on a complex multi-player game often stalls: random initial priors
give MCTS no traction, all games end with uniform scores, and the value head sees no signal.
Behavioral cloning (BC) from a pretrained MaskablePPO policy warm-starts the AlphaZeroNetwork
with a delivery-aware prior so that MCTS immediately finds scoring lines within its simulation
budget.

---

## Key Architectural Finding

`BusMaskableActorCriticPolicy` (in `rl/policies.py`) is a **single flat output head** — a standard
SB3 `MaskableActorCriticPolicy` with logit clamping added. There are **no per-phase output layers**
in the PPO network. Phase routing is entirely handled by the environment through action masking.

The per-phase heads described in `mcts_plan.md` are a feature of `AlphaZeroNetwork` only
(`use_per_phase_heads=True`), not the PPO model. The `HeadUsageCallback` in `rl/callbacks.py`
tracks which game phase was active for TensorBoard logging only — it has no effect on the network.

BC distillation from the flat PPO distribution to a per-phase AlphaZeroNetwork is straightforward:
trim the flat `(max_head_actions,)` PPO probability vector to the active head's size and renormalize.

---

## MCTS Action Masking Verification

**Both flat and per-phase MCTS modes are correct.**

- `env.action_masks()` returns `(max_head_actions,)` where True entries are always head-local
  indices in `[0, head_size_current_head)`.
- In flat mode: `priors` shape `(max_head_actions,)` — valid `action_idx` always in range.
- In per-phase mode: `priors` shape `(head_size,)` — valid `action_idx` < `head_size = len(priors)`.
  The guard `if action_idx < len(priors) else 0.0` in `MCTSNode.expand()` is a safety check
  that never triggers in a correct environment.
- Inside `AlphaZeroNetwork.forward()`, the mask is trimmed to the active head's logit width
  before being applied — correct for per-phase mode.

---

## Bootstrap Pipeline

### Step 0: Prerequisites

- Trained PPO model: `logs/ppo_bus_{RUNNAME}/best_pool_model/best_model.zip`
- Trained with `--use-slot-actionability` (required to match obs_dim)

### Step 1: Collect BC Game Records

```bash
python scripts/collect_bc_data.py \
    --ppo_checkpoint logs/ppo_bus_{RUNNAME}/best_pool_model/best_model.zip \
    --n_games 1000 \
    --output logs/bc_data/bc_data.pkl \
    --use_slot_actionability \
    --num_players 4 \
    --device auto
```

Produces: `logs/bc_data/bc_data.pkl` — a pickle of `list[dict]` with one record per step:
```
obs          : np.ndarray (obs_dim,)       — environment observation
head_id      : int                         — active HeadId (0–9)
action_probs : np.ndarray (max_head_actions,) — masked PPO probability distribution
mask         : np.ndarray (max_head_actions,) — boolean action mask
value_target : np.ndarray (num_players,)  — rank-based z from game outcome
player_id    : int
game_id      : int
move_number  : int
```

### Step 2: BC Pre-train AlphaZeroNetwork

```bash
python scripts/bc_pretrain.py \
    --bc_data logs/bc_data/bc_data.pkl \
    --output logs/bc_warmstart/network.pt \
    --epochs 10 \
    --batch_size 512 \
    --lr 1e-3 \
    --trunk_layers 512 512 256 \
    --use_per_phase_heads \
    --use_slot_actionability \
    --num_players 4 \
    --device auto
```

Produces: `logs/bc_warmstart/network.pt` + `logs/bc_warmstart/network_config.json`
Compatible with `--initial_checkpoint` in `scripts/train_mcts.py`.

**Policy loss**: soft cross-entropy between AlphaZeroNetwork logits and PPO probability distribution.
For per-phase heads: PPO probs are trimmed to `head_size` and renormalized per head.
**Value loss**: MSE between value head output and rank vector z.

### Step 3: AlphaZero Run 1 (BC-warmed)

```bash
python scripts/train_mcts.py \
    --iterations 100 \
    --games_per_iter 25 \
    --n_simulations 50 \
    --use_slot_actionability \
    --use_per_phase_heads \
    --n_workers 1 \
    --trunk_layers 512 512 256 \
    --train_steps 500 \
    --batch_size 512 \
    --replay_buffer_size 50000 \
    --min_buffer_size 2500 \
    --lr 5e-4 \
    --initial_checkpoint logs/bc_warmstart/network.pt \
    --checkpoint_dir logs/alphazero_bc_run1 \
    --device auto \
    --use_reward_shaping \
    --tensorboard
```

**Why `lr=5e-4` instead of `1e-3`?** The BC-warmed network already has a useful prior; a lower
learning rate prevents early low-quality MCTS iterations from overwriting BC knowledge.
**Why `--use_reward_shaping`?** Provides within-game differentiation during the transition period
before MCTS produces high-quality z targets.

### Step 4: AlphaZero Ramp-up and Full Strength

After Run 1 stabilizes (eval/avg_rank > 0.52 for several consecutive evaluations), continue
with the runs defined in `bash_scripts/alphazero_bootstrap.bash`:

- **Run 2**: `n_simulations=200`, `games_per_iter=50`, `lr=3e-4`
- **Run 3**: `n_simulations=400`, `games_per_iter=100`, `lr=2e-4`

Each run passes `--initial_checkpoint` to the previous run's `incumbent.pt`.

---

## Config Consistency Rules

When passing `--initial_checkpoint` to `train_mcts.py`, the following must match:
- `--use_per_phase_heads` must be set if BC used it (embedded in `_config.json`)
- `--trunk_layers` must match
- `--use_slot_actionability` must match (affects obs_dim)
- `--num_players` must match

`train_mcts.py` will print a warning if any mismatch is detected at startup.

---

## Smoke Tests

```bash
# Test BC collection (5 games, fast)
python scripts/collect_bc_data.py \
    --ppo_checkpoint logs/ppo_bus_{RUNNAME}/best_pool_model/best_model.zip \
    --n_games 5 --output /tmp/bc_smoke.pkl --use_slot_actionability --device cpu

# Test BC pretraining (2 epochs, small batch)
python scripts/bc_pretrain.py \
    --bc_data /tmp/bc_smoke.pkl --output /tmp/bc_net.pt \
    --epochs 2 --batch_size 32 --use_per_phase_heads --use_slot_actionability --device cpu

# Test MCTS with BC checkpoint (1 iteration, 5 sims)
python scripts/train_mcts.py \
    --iterations 1 --games_per_iter 2 --n_simulations 5 \
    --use_slot_actionability --use_per_phase_heads \
    --n_workers 1 --train_steps 5 --batch_size 16 \
    --replay_buffer_size 500 --min_buffer_size 1 \
    --checkpoint_dir /tmp/az_smoke \
    --initial_checkpoint /tmp/bc_net.pt --device cpu
```

See `bash_scripts/alphazero_bootstrap.bash` for the full phased execution script.
