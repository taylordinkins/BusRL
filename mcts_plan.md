# AlphaZero-Style MCTS Implementation Plan for Bus

## Context

The project currently uses MaskablePPO (via sb3-contrib) for training, with `rl/mcts.py` and `rl/mcts_player.py` existing as **inference-time-only** placeholders — they augment a pre-trained PPO model with search but have no training loop and do not correctly handle multi-player value backpropagation. This plan replaces those files entirely and builds a proper AlphaZero-style system:

- MCTS as **both training mechanism and inference enhancement**
- A standalone `AlphaZeroNetwork` (shared trunk + policy head + value head), not tied to SB3's MaskablePPO
- A dedicated training script (`scripts/train_mcts.py`) that keeps this completely separate from the existing MaskablePPO pipeline
- Correct multi-player MCTS backpropagation using per-player value tracking
- Rank-based value targets (1st place = 1.0, last = 0.0, linear interpolation)
- Terminal-outcome-only training signal by default (pure AlphaZero), with a flag to re-enable reward shaping
- Policy head as either flat 1670-logit OR per-phase (10 heads), switchable via argument. Both need to handle the use_slot_actionability flag

NOTE: 1670 may not be fixed, given the use_slot_actionability flag

---

## Architecture Overview

```
AlphaZeroNetwork
├── Trunk MLP (shared)          obs → latent features
├── Policy Head                 latent → action logits
│   ├── Flat mode: Linear(latent, 1670)
│   └── Per-phase mode: ModuleList of 10 Linear layers sized per HeadId
└── Value Head                  latent → rank estimates for all players, shape (num_players,), each ∈ [0, 1]

Training Loop (AlphaZeroTrainer)
├── Phase 1 - Self-play:        run N games with MCTS, collect (obs, π_mcts, z) samples
├── Phase 2 - Training:         optimize L = L_policy + L_value + L2_reg
└── Phase 3 - Evaluation:       new model vs previous checkpoint; promote if better
```

---

## Detailed Design

### 1. Multi-Player Value Formulation

Since Bus is a perfect-information game, the value network outputs a **vector of shape `(num_players,)`** — one expected rank per player — from the shared observation. Training target `z` is the full rank vector computed from the final game outcome:

```
z[i] = (num_players - rank[i]) / (num_players - 1)   # e.g., 4 players: 1st→1.0, 4th→0.0
```

Ties: averaged rank (e.g., two players tied for 1st → rank 1.5 → z ≈ (4 - 1.5) / 3 ≈ 0.833).

**Per-player value tracking in MCTS nodes:**
Each node stores `value_sum[num_players]` (one per player) and a single `visit_count`. On backpropagation, every simulation contributes to **all** player slots:
- At **terminal** nodes: ranks of ALL players are known → propagate each player's rank to their respective slot in every ancestor
- At **non-terminal** leaf nodes: value network predicts `(num_players,)` rank estimates → propagate all slots simultaneously to every ancestor
- When selecting children (PUCT): use `value_sum[current_player] / visit_count` as Q

This is cleaner than the scalar approach — every simulation is fully informative for all players, with no partially-filled slots.

### 2. Policy Head Design

Controlled by `use_per_phase_heads: bool` in `AlphaZeroNetworkConfig`.

**Flat mode** (`use_per_phase_heads=False`):
- Output: `Linear(trunk_dim, 1670)` — all action logits at once
- Masking: apply action mask in-place before softmax
- Policy target stored in self-play: shape `(1670,)` visit distribution

**Per-phase mode** (`use_per_phase_heads=True`):
- Output: `ModuleList` of 10 `Linear(trunk_dim, head_size[i])` layers
- `head_size[i]` read from `HierarchicalActionMapping.get_head_catalog_size(head_id)`
- Active head selected by `head_id` from the observation's global features (already encoded as 10-hot)
- Masking: apply within the active head's subspace
- Policy target stored in self-play: shape `(head_size[active_head],)` visit distribution

**Both modes** respect existing `action_masks()` from `BusEnv`.

---

## Files to Create

### `rl/alphazero_network.py`

```python
@dataclass
class AlphaZeroNetworkConfig:
    obs_dim: int                    # from ObservationConfig.total_observation_dim
    total_actions: int = 1670       # from ActionSpaceConfig.total_actions
    trunk_layers: list[int]         # e.g. [512, 512, 256]
    use_per_phase_heads: bool = False
    value_output: str = "sigmoid"   # sigmoid → [0,1] rank target
    trunk_activation: str = "relu"
    use_layer_norm: bool = True

class AlphaZeroNetwork(nn.Module):
    - __init__(config, head_catalog_sizes: dict[int, int])
    - forward(obs) → (policy_logits_or_dict, value)
    - get_policy_logits(features, head_id, mask) → logits for active head
    - get_value(features) → np.ndarray  # shape (num_players,)
    - save(path), load(path)  # custom .pt format, no SB3 dependency
```

Key details:
- Trunk uses `nn.LayerNorm` after each linear if `use_layer_norm=True`
- Value head: `Linear(trunk_dim, num_players) + Sigmoid` → output shape `(num_players,)`, each ∈ [0, 1]
- Per-phase heads: `nn.ModuleList` indexed by `HeadId` int value
- `head_catalog_sizes` sourced from `HierarchicalActionMapping` at construction time

---

### `rl/mcts.py` — Full rewrite (replaces placeholder)

```python
@dataclass
class MCTSConfig:
    n_simulations: int = 400
    c_puct: float = 1.5
    temperature: float = 1.0         # used during self-play data collection
    temperature_threshold: int = 30  # use temp=1.0 for first N moves, then greedy
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25  # set 0 for evaluation
    num_players: int = 4

class MCTSNode:
    - env: BusEnv (cloned)
    - parent, action_idx, prior
    - visit_count: int
    - value_sum: np.ndarray  # shape (num_players,) — tracks per-player values
    - children: dict[int, MCTSNode]
    - current_player: int   # cached from env
    - is_expanded, is_terminal

    Properties:
    - q_value(player_id) → value_sum[player_id] / visit_count
    - ucb_score(c_puct, current_player, q_min, q_max) → Q_norm(current_player) + c_puct * P * sqrt(N_p) / (1+N)
      where Q_norm = (Q - q_min) / (q_max - q_min + 1e-8)  # min-max over siblings (PUCT value centering)

    Methods:
    - expand(priors)
    - backpropagate(values: np.ndarray)  # shape (num_players,); fills all slots every simulation
    - select_child(c_puct, current_player_id) → MCTSNode
      # computes q_min/q_max across all children before calling ucb_score

class AlphaZeroMCTS:
    - __init__(network: AlphaZeroNetwork, config: MCTSConfig)
    - search(env) → int  # best action (greedy by visit count)
    - search_with_policy(env) → (action, visit_distribution)  # for data collection
    - _get_priors_and_value(env) → (priors_array, values_all_players: np.ndarray)  # shape (num_players,)
    - _get_terminal_values(env) → np.ndarray  # shape (num_players,), ranks [0,1]
    - _add_dirichlet_noise(priors, mask) → priors
    - _select_action(root, move_number) → int  # temperature-aware
```

**Masking invariants:**
- Always call `env.action_masks()` before expanding a node
- Mask logits to `-1e9` (not `-inf`) before softmax when computing priors — avoids NaN gradients during backprop while still zeroing masked probabilities after softmax
- In flat mode: apply a boolean mask over the full 1670-dim logit tensor (`logits[~mask] = -1e9`) before softmax; the resulting prior array is the same shape but invalid actions have ~0 probability
- In per-phase mode: apply the mask within the active head's subspace only; same `-1e9` rule applies
- Never create child nodes for masked actions
- Policy cross-entropy loss uses `log(p_θ + 1e-9)` to guard against log(0) when valid actions have zero MCTS visits

---

### `rl/alphazero_self_play.py`

```python
@dataclass
class SelfPlaySample:
    obs: np.ndarray              # shape (obs_dim,)
    head_id: int                 # active HeadId value
    policy_target: np.ndarray    # MCTS visit distribution (flat 1670, or head-local)
    value_target: np.ndarray     # z: rank vector for all players, shape (num_players,), each ∈ [0, 1]
    player_id: int
    game_id: int
    move_number: int

class ReplayBuffer:
    - __init__(max_size: int)
    - add_game(samples: list[SelfPlaySample])
    - sample_batch(batch_size: int) → list[SelfPlaySample]
    - __len__
    - save(path), load(path)

class SelfPlayWorker:
    - __init__(network, env_factory, mcts_config)
    - play_game() → list[SelfPlaySample]
      # Runs one complete game; stores (obs, π_mcts, player_id) per move
      # At game end: compute rank targets for each move's acting player
      # Returns labeled SelfPlaySamples

def run_self_play_parallel(
    network, env_factory, mcts_config, n_games, n_workers
) → list[SelfPlaySample]:
    # Uses concurrent.futures (ProcessPoolExecutor for game simulation,
    # ThreadPoolExecutor acceptable since torch releases GIL)
    # Returns flattened list of samples from all games
```

---

### `rl/alphazero_trainer.py`

```python
@dataclass
class AlphaZeroTrainingConfig:
    # Self-play
    games_per_iteration: int = 100
    n_self_play_workers: int = 4
    # Replay buffer
    replay_buffer_size: int = 200_000   # capacity in individual move-samples
    min_buffer_size: int = 10_000
    # Training
    train_steps_per_iteration: int = 1000
    batch_size: int = 512
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0          # gradient clipping threshold
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    # Evaluation
    eval_games: int = 20
    eval_rank_threshold: float = 0.55   # promote new model if avg norm rank exceeds this vs previous checkpoint
    eval_use_pool_avg: bool = False      # if True, promote based on avg rank vs checkpoint pool instead
    eval_pool_size: int = 10            # number of past checkpoints to retain in pool
    track_openskill: bool = False       # if True, maintain Openskill ratings across checkpoint pool
    # Checkpointing
    checkpoint_dir: str = "logs/alphazero"
    save_every_n_iterations: int = 5
    # Reward
    use_reward_shaping: bool = False    # False = terminal outcome only (default)

class AlphaZeroTrainer:
    - __init__(network_config, training_config, env_factory)
    - train(n_iterations: int)
    - _self_play_phase() → list[SelfPlaySample]
    - _training_phase(buffer) → dict  # loss breakdown
    - _evaluation_phase(new_net, old_net) → float  # avg normalized rank of new_net (higher = better)
    - _compute_loss(batch) → (policy_loss, value_loss)
    - save_checkpoint(iteration), load_checkpoint(path)
```

**Loss computation:**
```
L_policy = mean(-sum(π_mcts * log(p_θ + 1e-9)))  # cross-entropy vs MCTS visit distribution; 1e-9 prevents log(0)
L_value  = mean(||v_θ - z||²)                     # MSE vs rank vector z ∈ [0,1]^num_players
L_total  = policy_loss_weight * L_policy
         + value_loss_weight  * L_value
         + weight_decay * ||θ||²

# After loss.backward(): clip_grad_norm_(network.parameters(), max_grad_norm)
```

---

### `scripts/train_mcts.py`

Key CLI arguments:
```
--iterations N              number of self-play + train cycles (default: 100)
--games_per_iter N          games per self-play phase (default: 100)
--n_simulations N           MCTS simulations per move (default: 400)
--n_workers N               parallel self-play workers (default: 4)
--trunk_layers 512 512 256  hidden layer sizes for shared trunk
--use_per_phase_heads       flag: use per-phase policy heads (default: flat)
--use_reward_shaping        flag: enable step-level reward shaping (default: off)
--initial_checkpoint PATH   existing AlphaZeroNetwork .pt to resume from
--checkpoint_dir DIR        output directory (default: logs/alphazero)
--device auto|cpu|cuda
--num_players N             number of players (default: 4)
--eval_every N              evaluate every N iterations (default: 5)
--c_puct FLOAT              MCTS exploration constant (default: 1.5)
--dirichlet_alpha FLOAT     Dirichlet noise alpha (default: 0.3)
--dirichlet_epsilon FLOAT   Dirichlet noise weight at root (default: 0.25)
--temperature_threshold N   moves before switching to greedy selection (default: 30)
--lr FLOAT                  learning rate (default: 1e-3)
--max_grad_norm FLOAT       gradient clipping threshold (default: 1.0)
--batch_size N              training batch size (default: 512)
--replay_buffer_size N      max replay buffer size in move-samples (default: 200000)
--eval_use_pool_avg         flag: promote based on avg rank vs checkpoint pool (default: vs previous checkpoint)
--eval_pool_size N          number of past checkpoints to keep in pool (default: 10)
--track_openskill           flag: maintain Openskill ratings across checkpoint pool
--tensorboard               enable TensorBoard logging
```

Script flow:
1. Parse args → build `AlphaZeroNetworkConfig` + `AlphaZeroTrainingConfig`
2. Construct `AlphaZeroNetwork` (or load from `--initial_checkpoint`)
3. Construct `AlphaZeroTrainer`
4. Call `trainer.train(n_iterations)`

---

## Files to Modify

### `rl/mcts_player.py` — Full rewrite

Remove old `MCTSPlayer`, `PolicyPlayer`, `compare_players`. Replace with:

```python
class AlphaZeroPlayer:
    """Wraps AlphaZeroNetwork + AlphaZeroMCTS for GUI/evaluation use."""
    - __init__(network: AlphaZeroNetwork, mcts_config: MCTSConfig)
    - get_action(env) → int
    - get_action_with_stats(env) → (int, dict)

class AlphaZeroPolicyPlayer:
    """Direct network inference without MCTS (fast baseline)."""
    - __init__(network: AlphaZeroNetwork)
    - get_action(env) → int
```

Both expose the same `get_action(env) → int` interface for GUI compatibility.

### `rl/bus_env.py` — Minor verification

- Confirm `get_current_player()` exists and returns correct `int` for all game phases
- If missing, add it: reads from `self._engine.state.current_player_id` (or equivalent)
- Confirm `clone()` (line ~815) deep-copies all mutable state correctly (already looks complete)
- No reward-function changes needed; MCTS trainer computes value targets from ranks directly

### `rl/__init__.py` — Add exports

```python
from .alphazero_network import AlphaZeroNetwork, AlphaZeroNetworkConfig
from .mcts import AlphaZeroMCTS, MCTSConfig
from .mcts_player import AlphaZeroPlayer, AlphaZeroPolicyPlayer
from .alphazero_self_play import SelfPlaySample, ReplayBuffer, SelfPlayWorker
from .alphazero_trainer import AlphaZeroTrainer, AlphaZeroTrainingConfig
```

### `gui/` — GUI integration (after core is functional)

- `gui/dialogs.py`: add `AlphaZeroPlayer` as a selectable player type in `NewGameDialog`
- `gui/game_controller.py`: load AlphaZero models via `AlphaZeroNetwork.load(path)` + `AlphaZeroPlayer(network, config)`
- No interface change needed — both old `PolicyPlayer` and new `AlphaZeroPlayer` use `get_action(env)`

---

## Files to Remove / Replace

| File | Action | Reason |
|------|--------|--------|
| `rl/mcts.py` | **Replace entirely** | Inference-only placeholder; wrong multi-player backprop |
| `rl/mcts_player.py` | **Replace entirely** | Coupled to MaskablePPO; no training support |

New versions keep the same filenames (`rl/mcts.py`, `rl/mcts_player.py`) with entirely new content.

---

## Implementation Order

**Phase 1 — Network (standalone, testable first)**
1. `rl/alphazero_network.py`
2. Verify: forward pass produces correct output shapes for both head modes; mask application doesn't leak gradients

**Phase 2 — MCTS Core**
3. `rl/mcts.py` (new `AlphaZeroMCTS`)
4. Verify: run 10 simulations on a fresh `BusEnv`; every returned action passes `env.action_masks()`

**Phase 3 — Self-Play Pipeline**
5. `rl/alphazero_self_play.py`
6. Verify: single-worker game produces samples with correct shapes; value targets are all ∈ [0, 1]

**Phase 4 — Training Loop**
7. `rl/alphazero_trainer.py`
8. Verify: one iteration of self-play + training runs without error; policy and value losses both decrease over steps

**Phase 5 — Training Script**
9. `scripts/train_mcts.py`
10. Verify: smoke test completes end-to-end (see Verification section below)

**Phase 6 — Player Wrappers + Exports**
11. `rl/mcts_player.py` (new `AlphaZeroPlayer`, `AlphaZeroPolicyPlayer`)
12. `rl/__init__.py` (add exports)

**Phase 7 — GUI integration** (optional, after core is validated)
13. `gui/dialogs.py`, `gui/game_controller.py`

---

## Key Design Notes

### PUCT Value Centering
Raw Q-values across sibling nodes can vary widely in scale, especially early in training, causing the prior term to be either dominated or swamped. Before computing UCB scores in `select_child`, compute `q_min` and `q_max` across all children (for `current_player`), then normalize:

```
Q_norm = (Q - q_min) / (q_max - q_min + 1e-8)
PUCT   = Q_norm + c_puct * P * sqrt(N_parent) / (1 + N_child)
```

If a node has never been visited (`visit_count == 0`), treat Q as 0 before normalization. If all siblings have the same Q (or only one child exists), the denominator collapses to `1e-8` and normalization is a no-op — this is correct behavior.

### Masking in Per-Phase Mode
`BusEnv.action_masks()` already returns a mask sized to the active head's action space (via the hierarchical env design). The active head's logit tensor is the same size, so the mask is applied directly. No index-translation needed at inference time.

### Evaluation Metric
Use **average normalized rank** (the same `z` formula used for training targets) as the promotion criterion, averaged over `eval_games` games. This is logged per iteration. For promotion against the previous checkpoint: if `avg_rank_new > eval_rank_threshold`, promote. If `eval_use_pool_avg=True`, the baseline is the mean rank of pool members instead of the single previous checkpoint.

**Checkpoint pool**: the last `eval_pool_size` promoted checkpoints are retained on disk. Pool members serve as optional evaluation opponents and as the Openskill rating pool.

**Openskill ratings** (`--track_openskill`): if enabled, each evaluation game updates Openskill ratings for all competing checkpoints in the pool. Ratings are saved alongside checkpoint metadata and logged to TensorBoard if enabled. Requires the `openskill` Python package.

### Value Target Computation (no reward shaping)
At game end, ranks are computed from final scores (accounting for ties). Each stored sample's `value_target` is the full rank vector `z` of shape `(num_players,)` — the same vector for every sample from the same game, since final ranks are known for all players simultaneously. The `RewardCalculator` is not used for training targets — only `env._engine.state` scores at termination.

If `--use_reward_shaping` is set, the accumulated step rewards (from `BusEnv.step()`) are discounted and added to `z`. This is opt-in and off by default.

### Checkpoint Compatibility
`AlphaZeroNetwork` saves as `{name}.pt` (torch state dict) + `{name}_config.json`. This is **separate** from the MaskablePPO `.zip` format. Both training pipelines (`train.py` and `train_mcts.py`) coexist and produce independently loadable models. The `--initial_checkpoint` flag on `train_mcts.py` accepts an `AlphaZeroNetwork .pt` file to resume from (not a PPO checkpoint).

### `clone()` Performance
Each MCTS simulation calls `env.clone()`, which deep-copies `GameState`. At 400 simulations per move, this is the primary bottleneck. If profiling shows it is a bottleneck, a custom `__copy__` on `GameState`'s hot-path objects can reduce allocation overhead. For initial implementation, `copy.deepcopy` is correct and sufficient.

---

## Verification

1. **Smoke test** — runs end-to-end without error:
   ```
   python scripts/train_mcts.py --iterations 2 --games_per_iter 2 --n_simulations 20 --n_workers 1
   ```

2. **MCTS action validity** — every action returned by `AlphaZeroMCTS.search()` satisfies `env.action_masks()[action] == True`

3. **Value target range** — after one full game, all `SelfPlaySample.value_target` arrays have shape `(num_players,)` with all entries ∈ [0, 1]; the winning player's index averages higher than losing players' across samples

4. **Per-phase vs flat parity** — run the same game seed with both head modes; every step produces a legal action in both cases

5. **Mask integrity** — assert in `MCTSNode.expand()` that no masked action gets a child node; run 10 games without assertion errors

6. **PPO regression** — `scripts/train.py` still runs successfully after `rl/mcts.py` is replaced

---

## Implementation Checklist

- [x] Add checklist to mcts_plan.md
- [x] Fix `rl/bus_env.py` `clone()` to copy `_vrroomm_stage_state` and `_step_count`
- [x] Create `rl/alphazero_network.py` — `AlphaZeroNetworkConfig`, `AlphaZeroNetwork`
- [x] Rewrite `rl/mcts.py` — new `MCTSConfig`, `MCTSNode` (per-player value_sum), `AlphaZeroMCTS`
- [x] Create `rl/alphazero_self_play.py` — `SelfPlaySample`, `ReplayBuffer`, `SelfPlayWorker`, `run_self_play_parallel`
- [x] Create `rl/alphazero_trainer.py` — `AlphaZeroTrainingConfig`, `AlphaZeroTrainer`
- [x] Create `scripts/train_mcts.py` — CLI training script
- [x] Rewrite `rl/mcts_player.py` — `AlphaZeroPlayer`, `AlphaZeroPolicyPlayer`
- [x] Update `rl/__init__.py` — add AlphaZero exports

---

## Future Optimizations (not in initial implementation)

### Batched Network Evaluation during MCTS
The initial implementation calls the network once per simulation at leaf expansion (up to 400 serial forward passes per move). A significant speedup is possible by restructuring the search loop into three phases per simulation batch:
1. **Tree walk** — traverse from root to leaves for all simulations in the batch, recording the path
2. **Batch evaluate** — call the network once with all leaf observations stacked → single forward pass → `(batch, num_players)` value outputs + policy logits
3. **Expand + backprop** — expand and backpropagate in reverse for all simulations

This requires tracking virtual losses during the walk to prevent all simulations from collapsing to the same path. Defer until serial implementation is validated and profiling confirms this is the bottleneck.

---

## TODO: PPO Bootstrap (Option 2 — if AlphaZero cold-start stalls)

**When to consider this**: If `eval/avg_rank` does not trend above 0.50–0.51 by iteration 30 of Run 1
despite the 1a/1b fixes, it means the value head is still receiving insufficient differentiated signal
and MCTS is operating near-randomly. In that case, pre-warming AlphaZeroNetwork with a PPO-trained
policy via behavioral cloning (BC) is the recommended next step.

**Why direct weight transfer won't work**: SB3's MaskablePPO uses a separate `FlattenExtractor` trunk,
a single-player critic, and no LayerNorm — architecture is incompatible with AlphaZeroNetwork's shared
trunk + multi-player value head. BC sidesteps this by treating the PPO policy purely as a data source.

**Procedure**:

1. **Train MaskablePPO** using `scripts/train.py` until the policy can reliably deliver passengers
   (watch for mean episode reward rising above 0 in training logs). A few million timesteps is usually
   sufficient to get a policy that occasionally delivers.

2. **Collect BC game records**: Write a script that loads the trained SB3 model and runs N games
   (suggested: 500–2000) using the PPO policy greedily (argmax of masked logits). For each step record:
   - `obs` (the current observation)
   - `head_id` (from `env._get_decision_context()`)
   - `action_probs` (masked softmax of PPO logits — use `ppo.policy.get_distribution(obs_t, action_masks=mask_t).distribution.probs`)
   - At game end, compute `z` via `_compute_rank_vector(env)` from `rl/alphazero_self_play.py` and
     attach it to every step in that game.

3. **Supervised pre-training of AlphaZeroNetwork**: Train a fresh AlphaZeroNetwork on the collected
   records for a few epochs:
   - Policy loss: `F.cross_entropy(logits, ppo_probs)` — teach the trunk to reproduce PPO's masked distribution
   - Value loss: `F.mse_loss(value, z_targets)` — teach the value head from PPO-generated game outcomes
   - Use `lr=1e-3`, `weight_decay=1e-4`, `max_grad_norm=1.0` (same as Run 1 hyperparameters)
   - Stop when policy loss plateaus (usually 5–15 epochs over the dataset)

4. **Hand off to AlphaZero**: Save the BC-warmed network and pass it to Run 1 via
   `--initial_checkpoint logs/bc_warmstart/network.pt`. MCTS on top of a delivery-aware policy will
   find scoring lines within the simulation budget and immediately produce differentiated z targets.

**Key files involved**:
- `rl/alphazero_self_play.py` — `_compute_rank_vector` for computing z from PPO game records
- `rl/alphazero_network.py` — `AlphaZeroNetwork.save/load` for checkpoint hand-off
- `scripts/train.py` — existing PPO training pipeline
- `scripts/train_mcts.py` — Run 1 entry point; accepts `--initial_checkpoint`

**Note on `--use_reward_shaping` during BC warm-start**: Once the BC checkpoint is loaded, keep
`--use_reward_shaping` enabled for Run 1. The shaped z blend (alpha=0.15) is complementary — it
provides within-game differentiation even in games where the now-trained policy doesn't deliver.
