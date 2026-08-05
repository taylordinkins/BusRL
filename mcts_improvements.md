# MCTS Improvement Suggestions for BusRL

This document collects all suggestions surfaced during the Phase 2 AlphaZero log review.
Each item describes the problem it solves, where to make the change, and how to tune it.
Items are grouped by category and roughly ordered within each group by expected impact.

---

## Root Cause Summary

Two structural problems compound each other and explain why the agent replicates
the PPO failure mode (marker spam without deliveries):

1. **The value function cannot see deliveries.** A Vrroomm marker placed in
   `CHOOSING_ACTIONS` doesn't score until ~20–35 env steps later in
   `RESOLVING_ACTIONS`. With 50 MCTS simulations and a branching factor of ~8,
   the effective tree depth is `log₈(50) ≈ 1.9`. The MCTS tree never reaches the
   resolution phase from a placement decision, so the value signal at each leaf is
   pure network estimate — and the network hasn't learned that Vrroomm placements
   lead to deliveries yet.

2. **The value function trains on weak signal.** Pure terminal `z` (rank-based
   outcome with no reward shaping) gives the value head almost no gradient when
   games are close or all players fail to deliver. Until the value function learns
   to differentiate positions, the tree search is essentially guided by a noisy
   prior.

These two problems reinforce each other (chicken-and-egg). The fixes below address
both sides.

---

## Category 1: Bug Fixes

### 1.1 Eval reference checkpoint resets to BC warmstart on restart ✅ FIXED

**File:** `scripts/train_mcts.py:171`

**Problem:** On every restart, `trainer._prev_checkpoint_path` was unconditionally
set to `args.initial_checkpoint` (the BC warmstart). After the first promoted
incumbent, restarts compared the evolving network against the BC warmstart rather
than the best AlphaZero checkpoint. This makes promotion meaningless — a network
that has drifted away from BC but hasn't learned anything useful can still "beat"
the BC warmstart.

**Fix (already applied):** Check for `incumbent.pt` in the checkpoint directory.
If it exists, call `trainer.load_checkpoint(incumbent.pt)` instead, which restores
network weights, optimizer state, and sets the reference path correctly.

```python
# scripts/train_mcts.py
if args.initial_checkpoint:
    incumbent_path = Path(args.checkpoint_dir) / "incumbent.pt"
    if incumbent_path.exists():
        print(f"Incumbent found; resuming from {incumbent_path}.")
        trainer.load_checkpoint(str(incumbent_path))
    else:
        trainer._prev_checkpoint_path = args.initial_checkpoint
```

---

### 1.2 MCTS terminal value uses a different score than training `z` ✅ FIXED

**Files:** `rl/mcts.py:281`, `rl/alphazero_self_play.py:73`

**Problem:**
- `mcts.py:_get_terminal_values` ranks players by `p.score` (raw delivery count).
- `alphazero_self_play.py:_compute_rank_vector` ranks players by
  `p.get_final_score()` which returns `score - time_stones`.

In tight games where delivery counts are tied and only time stone penalties
separate players, the MCTS tree backpropagates values computed from one ranking
while the network is trained on labels computed from a different ranking. The
value head receives inconsistent supervision.

**Fix:** Change `mcts.py:282` to use `p.get_final_score()`.

```python
# rl/mcts.py — _get_terminal_values()
# Before:
scores = {p.player_id: p.score for p in state.players}
# After:
scores = {p.player_id: p.get_final_score() for p in state.players}
```

---

## Category 2: Observability

### 2.1 TensorBoard overlapping curves from multiple restarts ✅ FIXED

**File:** `rl/alphazero_trainer.py:99–104`

**Problem:** Every restart creates a new event file in the same `tb/` directory
with a step counter that resets to `iteration × train_steps`. TensorBoard reads
all files together and renders overlapping, sawtooth-shaped curves that make it
impossible to read any individual run.

**Fix:** Append a timestamp subdirectory to the `SummaryWriter` path so each
run gets its own named series.

```python
# rl/alphazero_trainer.py — inside __init__, tensorboard block
import datetime
run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
self._writer = SummaryWriter(
    log_dir=str(self._checkpoint_dir / "tb" / run_tag)
)
```

TensorBoard will then show each run as a separately named series that can be
toggled or overlaid intentionally.

---

### 2.2 Increase eval games from 10 to 30–50 ✅ APPLIED (bash script updated to 40)

**File:** `bash_scripts/alphazero_bootstrap.bash` Phase 2 command / `--eval_games`

**Problem:** With 10 eval games, the standard error of `eval/avg_rank` is
approximately `0.5 / sqrt(10) ≈ 0.16`. The difference between the promotion
threshold (0.52) and the random baseline (0.50) is smaller than the noise floor.
The observed values ranging from 0.38 to 0.70 are essentially all noise.

**Fix:** Use `--eval_games 40`. This brings the standard error down to ~0.08,
enough to detect genuine improvement. The extra time is worth paying.

---

### 2.3 Log delivery rate per self-play batch ✅ FIXED

**File:** `rl/alphazero_self_play.py:run_self_play_parallel`

**Problem:** There is no visibility into whether self-play games produce any
passenger deliveries. If every game ends at 0–0 (no deliveries), all z values
cluster near 0.5 regardless of reward shaping, and you can diagnose this
immediately. If games average 1–3 deliveries, the value head just needs more
iterations.

**Fix:** Log `max_score`, `min_score`, and mean score across games in the
`verbose` printout at the end of `run_self_play_parallel`. The existing
`all_scores` list is already populated; just add:

```python
# rl/alphazero_self_play.py — end of run_self_play_parallel verbose block
if all_scores:
    max_scores = [max(d.values()) for d in all_scores]
    print(
        f"  Deliveries per game: mean={sum(max_scores)/len(max_scores):.2f}"
        f"  min={min(max_scores)}  max={max(max_scores)}"
        f"  zero_delivery_games={sum(1 for s in max_scores if s == 0)}/{len(max_scores)}"
    )
```

Also log `train/max_score_mean` to TensorBoard from the trainer so it's visible
across iterations alongside policy/value loss.

---

## Category 3: Training Signal Quality

### 3.1 Enable reward shaping in Phase 2 (`--use_reward_shaping`) ✅ APPLIED (bash script updated)

**File:** `bash_scripts/alphazero_bootstrap.bash` Phase 2 command

**Problem:** The current Phase 2 command does not include `--use_reward_shaping`,
so the training `z` is pure terminal rank. `SelfPlayWorker.play_game()` already
implements shaped `z` blending when this flag is on (`alpha=0.15`, ranking players
by cumulative step rewards). The `delivery_reward` of `1.0` per delivery
accumulates in `cumulative_rewards` and produces a meaningfully stronger z for
players who delivered versus those who only placed markers.

**Fix:** Add `--use_reward_shaping` to the Phase 2 command. The alpha of 0.15 is
conservative enough that it doesn't overwhelm terminal signal. Keep it on through
Phase 3; consider disabling in Phase 4 once the value head is well-trained.

---

### 3.2 Add a policy KL regularization term to prevent BC erosion ✅ IMPLEMENTED

**File:** `rl/alphazero_trainer.py:_compute_loss_per_phase`

**Problem:** The policy loss has been trending upward (0.87 → 0.97 over 50 iters),
meaning the BC warmstart knowledge is eroding. At 50 sims, the MCTS visit
distribution is nearly uniform over legal actions, and training on these targets
overwrites the informative BC prior without replacing it with anything better.

**Fix:** Load the BC checkpoint once at trainer initialization and add a small
KL divergence from the current policy toward the BC policy as an auxiliary loss.
Decay the weight linearly over the run.

```python
# In AlphaZeroTrainer.__init__ (add optional bc_checkpoint_path parameter):
self._bc_network = None
if bc_checkpoint_path is not None:
    from .alphazero_network import AlphaZeroNetwork
    self._bc_network = AlphaZeroNetwork.load(bc_checkpoint_path)
    self._bc_network.to(self.device)
    self._bc_network.eval()

# In _compute_loss_per_phase, after computing log_probs:
if self._bc_network is not None and kl_weight > 0:
    with torch.no_grad():
        bc_logits, _ = self._bc_network(obs_t, head_id=head_id)
        bc_log_probs = F.log_softmax(bc_logits, dim=1)
    # KL(BC || current) — penalizes moving away from the BC prior
    kl_loss = F.kl_div(log_probs, bc_log_probs.exp(), reduction="batchmean")
    policy_loss = policy_loss + kl_weight * kl_loss
```

**Tuning:** Start `kl_weight=0.1` and decay linearly to `0.0` over the 100
iterations of Phase 2. Pass `--bc_checkpoint logs/bc_warmstart/network.pt` to
`train_mcts.py`. Once the policy loss stabilizes or starts decreasing on its own,
the BC anchor is no longer needed.

---

### 3.3 Reduce Dirichlet noise weight during the BC warmstart phase ✅ APPLIED (bash script: 0.25 → 0.15)

**File:** `bash_scripts/alphazero_bootstrap.bash` Phase 2 command

**Problem:** The current `--dirichlet_epsilon 0.25` injects 25% random Dirichlet
noise into the root policy prior. In the original AlphaZero this is calibrated for
a strong, already-converged policy. During a BC warmstart where the policy carries
real information about the game, 25% noise partially cancels the BC signal and
makes the root search more random than it needs to be.

**Fix:** Use `--dirichlet_epsilon 0.15` for Phase 2. Ramp back to 0.25 in
Phase 3 when the network has had more self-play experience.

---

## Category 4: MCTS Search Quality (the Tree Depth Problem)

### 4.1 Why increasing simulation count alone does not fix depth ~2

The expected tree depth with `N` simulations and branching factor `B` is
`log_B(N)`. During `CHOOSING_ACTIONS`, `B ≈ 8` (7 areas + PASS):

| Simulations | Effective tree depth |
|-------------|----------------------|
| 50          | 1.9                  |
| 200         | 2.5                  |
| 1,000       | 3.3                  |
| 10,000      | 4.4                  |
| ~10¹⁸       | 20.0 (sees a delivery) |

There is no simulation budget that makes the tree deep enough to see deliveries
through brute-force expansion. Increasing sims does improve policy target quality
marginally (visit distributions are less uniform), which justifies going to
100–150 sims in Phase 2, but it does not address the fundamental horizon problem.

---

### 4.2 Rollout-augmented leaf evaluation (primary fix for tree depth) ✅ IMPLEMENTED

**File:** `rl/mcts.py` — `MCTSConfig`, `AlphaZeroMCTS._simulate`,
new method `_get_value_with_rollout`

**Problem:** MCTS relies entirely on the value network to bootstrap the horizon
beyond the tree frontier. When the value network is untrained, shallow trees
provide no useful signal at all. The fix is to give each simulation free extra
depth by running a greedy policy rollout from the leaf before querying the value
network — the AlphaGo (pre-AlphaZero) approach.

With `rollout_steps=30`, each of 50 simulations now sees ~32 moves total.
From a mid-`CHOOSING_ACTIONS` leaf, 30 greedy policy steps reliably advance
through the rest of the round and into the next one — exactly where Vrroomm
deliveries appear. This replaces the chicken-and-egg dependency between tree depth
and value function quality.

**Changes to `MCTSConfig`:**

```python
@dataclass
class MCTSConfig:
    # ... existing fields ...
    rollout_steps: int = 0          # 0 = disabled; >0 = greedy steps from leaf
    rollout_reward_weight: float = 0.0  # blend weight: rollout rewards vs network value
```

**New method `_get_value_with_rollout`:**

```python
def _get_value_with_rollout(self, env: "BusEnv") -> np.ndarray:
    """Run a greedy policy rollout from a leaf, then query the value network.

    Accumulates per-player step rewards during the rollout and optionally
    blends them into the returned value vector.
    """
    rollout_env = env.clone()
    cumulative_rewards = np.zeros(self.config.num_players, dtype=np.float32)

    for _ in range(self.config.rollout_steps):
        if rollout_env._engine is None or rollout_env._engine.is_game_over():
            return self._get_terminal_values(rollout_env)

        priors, _ = self._get_priors_and_value(rollout_env)
        mask = rollout_env.action_masks()
        priors[~mask] = 0.0
        total = priors.sum()
        if total > 0:
            action = int(np.argmax(priors))
        else:
            action = int(np.where(mask)[0][0])

        current_player = rollout_env.get_current_player()
        _, reward, terminated, truncated, _ = rollout_env.step(action)
        cumulative_rewards[current_player] += float(reward)

        if terminated or truncated:
            return self._get_terminal_values(rollout_env)

    _, values = self._get_priors_and_value(rollout_env)

    if self.config.rollout_reward_weight > 0:
        n = self.config.num_players
        order = sorted(range(n), key=lambda p: -cumulative_rewards[p])
        reward_z = np.zeros(n, dtype=np.float32)
        for rank, pid in enumerate(order):
            reward_z[pid] = (n - (rank + 1)) / max(n - 1, 1)
        w = self.config.rollout_reward_weight
        values = (1 - w) * values + w * reward_z

    return np.clip(values, 0.0, 1.0)
```

**Change to `_simulate`** — replace the leaf evaluation block (lines 247–253):

```python
# Evaluation
if node.is_terminal:
    values = self._get_terminal_values(node.env)
else:
    priors, values = self._get_priors_and_value(node.env)
    if self.config.rollout_steps > 0 and not node.is_expanded:
        values = self._get_value_with_rollout(node.env)
    if not node.is_expanded:
        node.expand(priors)
```

Note: `_get_priors_and_value` is still called for the priors needed by
`node.expand()`; the rollout only replaces the value estimate at unexpanded
(first-visit) leaves where the network estimate is least reliable.

**Add CLI arguments to `train_mcts.py`:**

```python
parser.add_argument("--rollout_steps", type=int, default=0)
parser.add_argument("--rollout_reward_weight", type=float, default=0.0)
```

**Tuning:**

| Parameter | Phase 2 (warmup) | Phase 3 | Phase 4 (full) |
|-----------|------------------|---------|----------------|
| `rollout_steps` | 30 | 15 | 0 |
| `rollout_reward_weight` | 0.3 | 0.1 | 0.0 |

Taper both toward zero as the value function matures. Check `train/v_std` — once
it consistently exceeds ~0.35, the value head is differentiating positions well
enough to stand on its own.

**Performance cost:** Each simulation costs `rollout_steps` additional greedy
forward passes instead of 1. At `rollout_steps=30`, total network queries per move
go from 50 to ~50×31 = 1,550. On a GPU with batching, expect roughly 3–6× slower
self-play. The data quality improvement should more than compensate at the
scale of 25 games/iter.

---

### 4.3 "Roll to end of round" variant (alternative to fixed-K rollout) ✅ IMPLEMENTED

Rather than a fixed `rollout_steps`, roll forward until the game phase returns
to `CHOOSING_ACTIONS` (i.e., the round has fully resolved). This is semantically
cleaner — every leaf evaluation sees exactly one complete resolution cycle
regardless of where in the round the tree frontier happens to be.

**Stopping condition** (replace the step counter loop):

```python
from core.constants import Phase

initial_round = rollout_env._engine.state.global_state.round_number
for step in range(max_rollout_steps):  # safety cap ~60
    if rollout_env._engine is None or rollout_env._engine.is_game_over():
        return self._get_terminal_values(rollout_env)
    current_phase = rollout_env._engine.state.phase
    current_round = rollout_env._engine.state.global_state.round_number
    # Stop when we reach the start of the next round's choosing phase
    if (step > 0
            and current_phase == Phase.CHOOSING_ACTIONS
            and current_round > initial_round):
        break
    # ... action selection and step ...
```

Set `max_rollout_steps=60` as a safety cap against runaway loops.

---

### 4.4 `--mcts-use-rollout` in `evaluate.py` is currently a no-op

**File:** `scripts/evaluate.py:65`, `rl/mcts.py:MCTSConfig`

`evaluate.py` already passes `use_value_network=not args.mcts_use_rollout` to
`MCTSConfig`, but `MCTSConfig` does not have a `use_value_network` field, so
the argument is silently ignored. Once `rollout_steps` is added to `MCTSConfig`,
`evaluate.py` should be updated to pass `rollout_steps` instead.

---

## Category 5: Data and Curriculum

### 5.1 Mid-game curriculum (strongest long-term fix) ✅ IMPLEMENTED

**File:** `rl/alphazero_self_play.py:SelfPlayWorker.play_game`

**Problem:** MCTS trained from the very start of games spends most of its budget
learning the early-game marker placement strategy (round 1), where no deliveries
are yet possible. Round 1 is also the most variable in terms of board setup.

**Fix:** Play the first `k` rounds of each self-play game using the BC policy
(greedy, temperature 0) and only hand off to MCTS-based self-play after that.
This guarantees MCTS trains on states where Vrroomm deliveries are imminent and
the board is in a well-structured, realistic mid-game configuration.

```python
# In SelfPlayWorker.play_game(), before the main loop:
BC_ROUNDS = 1  # play round 1 with BC policy only

if self.bc_network is not None:
    while True:
        if env._engine is None or env._engine.is_game_over():
            terminated = True
            break
        round_num = env._engine.state.global_state.round_number
        if round_num > BC_ROUNDS:
            break
        # Greedy BC policy — no MCTS, no noise
        obs = env._get_observation()
        mask = env.action_masks()
        decision = env._get_decision_context()
        head_id_obj = decision.get("head_id") if decision else None
        head_id = head_id_obj.value if head_id_obj is not None else None
        action = self.bc_network.get_greedy_action(obs, mask, head_id)
        _, _, step_term, step_trunc, _ = env.step(action)
        if step_term or step_trunc:
            terminated = True
            break
    # MCTS self-play continues from here for rounds > BC_ROUNDS
```

Add `bc_network` as an optional parameter to `SelfPlayWorker.__init__` and
`run_self_play_parallel`. Set `BC_ROUNDS=1` initially; increase to 2 in Phase 3
if deliveries are still sparse.

---

### 5.2 Move-weighted replay buffer sampling (lower priority)

**File:** `rl/alphazero_self_play.py:ReplayBuffer.sample_batch`

**Problem:** All positions in a game are sampled with equal probability. Late-game
positions are far more informative for the value head (closer to terminal outcome)
and for the policy head (decisions are more consequential). Early-game positions,
especially from round 1, add noise without much signal.

**Fix:** Store `move_number` alongside each sample and apply inverse-distance
weighting: positions in the last 30% of the game are sampled 3× more often than
positions in the first 30%. This is a soft version of the curriculum in 5.1.

```python
def sample_batch(self, batch_size: int) -> list[SelfPlaySample]:
    n = min(batch_size, len(self._buffer))
    buf = list(self._buffer)
    max_move = max(s.move_number for s in buf) or 1
    weights = np.array(
        [(s.move_number / max_move) ** 0.5 + 0.1 for s in buf],
        dtype=np.float64,
    )
    weights /= weights.sum()
    indices = np.random.choice(len(buf), size=n, replace=False, p=weights)
    return [buf[i] for i in indices]
```

The `** 0.5` exponent gives a mild preference without completely discarding early
positions (which do matter for learning the setup phases).

---

## Summary Table

| # | Change | Files | Status | Expected impact |
|---|--------|-------|--------|----------------|
| 1.1 | Restart reference bug | `train_mcts.py` | ✅ Done | High (correctness) |
| 1.2 | Terminal score consistency | `mcts.py` | ✅ Done | Medium |
| 2.1 | TensorBoard per-run dirs | `alphazero_trainer.py` | ✅ Done | Low (observability) |
| 2.2 | Increase eval games to 40 | `alphazero_bootstrap.bash` | ✅ Done | High (correctness) |
| 2.3 | Log delivery rate | `alphazero_self_play.py` | ✅ Done | High (observability) |
| 3.1 | Enable reward shaping (Phase 2) | `alphazero_bootstrap.bash` | ✅ Done | High |
| 3.2 | KL regularization from BC | `alphazero_trainer.py`, `train_mcts.py` | ✅ Done | High |
| 3.3 | Reduce Dirichlet epsilon (Phase 2) | `alphazero_bootstrap.bash` | ✅ Done | Medium |
| 4.2 | Rollout-augmented leaf eval | `mcts.py`, `train_mcts.py` | ✅ Done | Very high |
| 4.3 | Roll-to-round-end variant | `mcts.py`, `train_mcts.py` | ✅ Done | Very high |
| 4.4 | Fix `evaluate.py` rollout flag | `evaluate.py` | Pending | Low (correctness) |
| 5.1 | Mid-game curriculum | `alphazero_self_play.py`, `train_mcts.py` | ✅ Done | Very high |
| 5.2 | Move-weighted replay sampling | `alphazero_self_play.py` | Pending | Medium |

---

## New CLI Arguments — Usage Guide

All new features are opt-in via `train_mcts.py` arguments. Nothing changes if you
don't pass them — existing runs are unaffected.

---

### Rollout-augmented leaf evaluation (`--rollout_steps`, `--rollout_reward_weight`, `--rollout_to_round_end`)

**What it does:** Instead of querying the value network immediately at an unexpanded
leaf, the MCTS first runs a greedy policy rollout for additional steps before
asking the network for a value estimate. This gives the tree an effective
extra 15–30 moves of horizon at no simulation budget cost.

**Mode 1 — Fixed-K rollout:**
```bash
--rollout_steps 30 --rollout_reward_weight 0.3
```
Runs exactly 30 greedy steps from each first-visit leaf. Use in Phase 2.
Taper to `--rollout_steps 15 --rollout_reward_weight 0.1` in Phase 3.
Drop entirely in Phase 4 once `train/v_std > 0.35`.

**Mode 2 — Roll to round end (recommended for Phase 2):**
```bash
--rollout_to_round_end --rollout_reward_weight 0.3
```
Rolls forward until the start of the next `CHOOSING_ACTIONS` phase (i.e., one
complete resolution cycle). `--rollout_steps` becomes a safety cap (default 60 if
not set). Semantically cleaner: every leaf evaluation sees exactly one delivery
window regardless of where in the round the tree frontier is.

**`--rollout_reward_weight`:** Blends rollout-accumulated delivery rank (0.3 = 30%)
with the network's value estimate (70%). When the value net is untrained, a higher
weight anchors the tree to real delivery signal. Taper toward 0.0 as the value
head matures.

---

### KL regularization from BC checkpoint (`--bc_checkpoint`, `--kl_weight`, `--kl_total_iters`)

**What it does:** Loads a frozen BC network at startup and adds a KL divergence
penalty to the policy loss during training: `KL(BC policy || current policy)`.
This anchors the current policy toward the BC prior and slows erosion of BC
knowledge during early low-quality MCTS samples.

```bash
--bc_checkpoint logs/bc_warmstart/network.pt \
--kl_weight 0.1 \
--kl_total_iters 100
```

The KL weight decays **linearly from `--kl_weight` to 0** over `--kl_total_iters`
iterations. After that point, the BC anchor is inactive and training is pure MCTS.

**When to use:** Phase 2 only. Set `--kl_weight 0.0` (the default) in Phase 3
and beyond once `policy_loss` has stabilized or is trending downward.

**Note:** `--bc_checkpoint` must point to an `AlphaZeroNetwork` `.pt` file (same
architecture as the training network). The same path is also used by the
mid-game curriculum (see below).

---

### Mid-game curriculum (`--bc_checkpoint`, `--bc_rounds`)

**What it does:** At the start of each self-play game, plays the first `--bc_rounds`
rounds using the BC policy (greedy, no MCTS, no training samples collected).
MCTS self-play begins from round `bc_rounds + 1`. This guarantees MCTS trains
on states where Vrroomm deliveries are imminent rather than wasting simulation
budget on the nearly-uniform early-game setup.

```bash
--bc_checkpoint logs/bc_warmstart/network.pt \
--bc_rounds 1
```

**Tuning schedule:**

| Phase | `--bc_rounds` |
|-------|--------------|
| Phase 2 | 1 |
| Phase 3 | 2 (if deliveries still sparse) |
| Phase 4 | 0 (omit entirely) |

**Note:** Requires `--bc_checkpoint`. If `--bc_rounds 0` (the default) this feature
is disabled even if `--bc_checkpoint` is provided (the BC network will still be
used for KL regularization if `--kl_weight > 0`).

---

### Delivery rate logging (automatic when `--self_play_verbose` or `verbose=True`)

No new flag needed. When self-play logging is enabled, the end-of-batch summary
now prints:

```
  Deliveries per game: mean=1.42  min=0  max=4  zero_delivery_games=3/25
```

Use this to diagnose whether MCTS is finding delivery lines at all. If
`zero_delivery_games` is near `n_games` through iteration 20+, the tree is still
failing to reach delivery states and the rollout / curriculum settings need
revisiting.
