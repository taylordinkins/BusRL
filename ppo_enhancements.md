# PPO Enhancement Plan: Vrroomm Delivery Improvement

**Based on run:** `ppo_bus_20260502_112313`
**Problem:** Policy has learned a good opening but consistently misses delivery (Vrroomm)
opportunities in mid-to-late rounds.

**Root cause summary:**
1. The policy cannot directly observe whether passengers are reachable or deliverable — it must implicitly infer network connectivity from 490 raw edge features, which becomes unreliable in complex late-game board states.
2. The eval is deterministic: all 10 eval episodes play the same game identically, so the eval curve is measuring N=1 not N=10.
3. The Vrroomm placement bonus is unconditional (+0.005 regardless of delivery viability) and 200× smaller than the delivery reward, providing a weak and noisy placement incentive.

**Obs dimension (current):** 1470 (default) / 1512 (with `use_slot_actionability`)

---

## Priority 1 — Observation Enrichment

### Motivation

The 5 current passenger features (`exists`, `location_node_idx`, `is_at_train_station`,
`is_at_central_park`, `is_at_matching_building`) give the policy no direct signal about
whether a passenger is reachable or deliverable. The policy must reconstruct rail-network
connectivity implicitly from the 70-edge feature block — a computation that an MLP handles
unreliably once the board has multiple overlapping networks in the late game.

Adding explicit reachability and delivery-viability flags per passenger, plus global
delivery-opportunity scalars, gives the policy direct first-order information for:

- Deciding whether to place a Vrroomm marker (CHOOSING_ACTIONS)
- Selecting the right passenger at stage 1 of resolution
- Selecting the right destination at stage 2

### 1.1 New Helper: `rl/delivery_utils.py`

Create a standalone module that computes delivery viability for a given player and building
type. The VrrooommResolver contains equivalent logic but it is not reusable outside resolution.

```
rl/delivery_utils.py
```

Public function:

```python
@dataclass
class DeliveryFeatures:
    passenger_reachable: dict[int, bool]       # passenger_id → on my network?
    passenger_valid_source: dict[int, bool]    # passenger_id → reachable AND not occupied?
    deliverable_count: int                     # valid sources with ≥1 destination
    available_slot_count: int                  # unoccupied matching building slots on my network


def compute_delivery_features(
    state: GameState,
    player_id: int,
    building_type: BuildingType,
) -> DeliveryFeatures:
```

**Algorithm inside `compute_delivery_features`:**

1. `network_nodes = state.board.get_player_network_nodes(player_id)` — called ONCE.

2. Compute `occupied_ids: set[int]` — replicate `VrrooommResolver._mark_existing_passengers_as_occupying`:
   - For each node with ≥1 building of `building_type`:
     - Sort passenger IDs at that node.
     - Pair them to matching slot indices (ascending). Each paired passenger ID enters `occupied_ids`.
   - This is correct during CHOOSING_ACTIONS (slot-level occupancy is always clear then)
     and provides a fallback during VRROOMM resolution.

3. Count `available_slot_count`:
   ```python
   sum(
       1 for nid in network_nodes
       for slot in state.board.get_node(nid).building_slots
       if slot.building == building_type
       and slot.occupied_by_passenger_id is None
   )
   ```
   `slot.occupied_by_passenger_id` is set by VrrooommResolver during resolution and cleared
   afterwards, so this correctly reflects the live slot state in both phases.

4. Per passenger: `is_reachable = passenger.location in network_nodes`.
   `is_valid_source`:
   - `is_reachable`
   - AND `passenger.passenger_id not in occupied_ids`
   - AND no slot at passenger's location has `occupied_by_passenger_id == passenger.passenger_id`
     (catches deliveries already made within the current VRROOMM slot)

5. A valid source is "deliverable" if `available_slot_count > 0` (assumes connected network —
   valid in virtually all cases after setup; document this assumption).

6. Return `DeliveryFeatures` with the above.

**Performance note:** `get_player_network_nodes` does a BFS over the player's edges. Call it
once per encoding, not per passenger. O(E + V) per call, negligible at 70 edges.

### 1.2 Per-Passenger Features: 5 → 7

Add two new features **after** the existing 5:

| Index | Feature | Notes |
|-------|---------|-------|
| 0 | `exists` | unchanged |
| 1 | `location_node_idx` (normalized) | unchanged |
| 2 | `is_at_train_station` | unchanged |
| 3 | `is_at_central_park` | unchanged |
| 4 | `is_at_matching_building` | unchanged (node has current-clock building; proxy for occupancy) |
| **5** | **`is_reachable_by_current_player`** | binary: `passenger.location in network_nodes` |
| **6** | **`is_valid_delivery_source`** | binary: reachable AND not slot-occupied |

Feature 6 directly tells the policy "I can pick up this passenger right now." Feature 5
is a looser signal useful when a passenger is on the network but all current-clock slots
are occupied (may become deliverable after clock advances or buildings are placed).

Both features are computed from the `DeliveryFeatures` struct returned by
`compute_delivery_features(state, current_player_id, current_clock_type)`.

### 1.3 Global Features: +3

Append to the existing 39 global features:

| Index (from offset) | Feature | Normalization |
|--------------------|---------|---------------|
| 39 | `my_deliverable_count_current` | `/ MAX_PASSENGERS` |
| 40 | `my_deliverable_count_next_clock` | `/ MAX_PASSENGERS` |
| 41 | `my_available_slots_current` | `/ (MAX_NODES * MAX_BUILDING_SLOTS_PER_NODE)` |

`my_deliverable_count_current`: `DeliveryFeatures.deliverable_count` for current clock type.

`my_deliverable_count_next_clock`: `DeliveryFeatures.deliverable_count` for the **next** clock
type in `TIME_CLOCK_ORDER`. This matters because:
- The clock auto-advances before Vrroomm resolves (unless an opponent stops it).
- At CHOOSING_ACTIONS time, the policy sees the current clock type but deliveries happen
  under the next type.
- An opponent with a TIME_CLOCK marker may stop the clock (visible from action board), but
  the next-clock feature lets the policy reason about the "advance" scenario regardless.

Compute by calling `compute_delivery_features` a second time with
`TIME_CLOCK_ORDER[(clock_idx + 1) % len(TIME_CLOCK_ORDER)]`.

`my_available_slots_current`: `DeliveryFeatures.available_slot_count` for current clock type.
Gives the policy a direct count of how "full" the delivery board is.

### 1.4 Train Station Connectivity (implicit)

The `is_at_train_station` per-passenger feature already encodes whether a passenger is
at a train station. The new `is_reachable_by_current_player` feature will show the policy
which train-station passengers (if any) are on its network. Combined with the global
`my_deliverable_count_next_clock`, the policy can infer "I have a train station connected,
passengers will likely spawn there, and there are delivery slots for the next clock type."
No additional explicit feature is needed for train station connectivity.

### 1.5 `config.py` Changes

`PASSENGER_FEATURE_DIM` is a `ClassVar` (not a dataclass field). Add:
- A new opt-in flag `use_delivery_features: bool = False`
- A `passenger_feature_dim` **property** that returns `PASSENGER_FEATURE_DIM + (2 if use_delivery_features else 0)`
- Update `passenger_features_size` property to use `passenger_feature_dim`
- Update `global_features_size` property to return `GLOBAL_FEATURE_DIM + (3 if use_delivery_features else 0)`
- Update `GLOBAL_FEATURE_DIM` doc comment accordingly

Following the same pattern as `use_slot_actionability`.

New obs dim with both flags enabled:
```
1470 + 15*2 + 3 = 1470 + 33 = 1503  (without use_slot_actionability)
1512 + 33 = 1545                     (with use_slot_actionability)
```

### 1.6 `observation.py` Changes

1. Import `compute_delivery_features` and `TIME_CLOCK_ORDER` at the top.

2. Add `current_player_id` parameter to `_encode_passengers`. Update the call in
   `encode()` to pass the already-computed `current_player_id`.

3. In `_encode_passengers`: if `config.use_delivery_features`:
   - Call `compute_delivery_features(state, current_player_id, current_clock_type)` once.
   - Call it again with the next clock type.
   - For each passenger, write features 5 and 6 from the first result's dicts.
   - The `feature_dim` loop must use `self.config.passenger_feature_dim` (not the ClassVar).

4. In `_encode_global`: if `config.use_delivery_features`:
   - Append the 3 global features after the existing 39 (at indices 39, 40, 41).

5. Replace all direct references to `self.config.PASSENGER_FEATURE_DIM` in the file with
   `self.config.passenger_feature_dim`.

### 1.7 `train.py` Changes

Add CLI flag (following `--use-slot-actionability` pattern):

```bash
--use-delivery-features
```

Pass to `ObservationConfig(use_delivery_features=args.use_delivery_features)`.

Print the obs dim when enabled.

### 1.8 Priming Note

This change alters the observation tensor shape. It is **incompatible with checkpoints
trained without this flag** (as noted for `use_slot_actionability`). A fresh priming run
is required:

```
--use-delivery-features
--use-slot-actionability          # keep if using this too
--load_pool_dir logs/ppo_bus_20260502_112313/opponent_pool
--start_fresh_directory
--prime_n_top 30
--prime_n_random 10
--initial_checkpoint logs/ppo_bus_20260502_112313/best_pool_model/best_model.zip
```

The existing model's weights are **not** loaded when the obs dim changes. The priming run
starts from a random init but with a filled pool, giving early diversity pressure.
Alternatively, reset only the actor head and transfer value/critic weights if SB3 supports
partial loading, but a clean restart is safer.

---

## Priority 2 — Stochastic Evaluation

### Motivation

The eval runs with `deterministic=True` in both `MaskableEvalCallback` and
`EvalStatsCallback`. With a deterministic policy, any fixed starting state produces the
same game trajectory every time. The 10 eval episodes have been returning identical rewards
for thousands of steps (confirmed from evaluations.npz: all 10 episodes return 9.138 in
the last 2M steps). This means the eval curve is N=1, not N=10 — useless for tracking
policy improvement.

The fix is stochastic evaluation: the policy samples from its distribution rather than
taking argmax. Different samples → different game trajectories → a genuine N=episode mean.

### 2.1 `train.py` Changes

Add CLI flag:
```bash
--eval_deterministic    # store_true; default is False (stochastic)
```

Pass to both callbacks:
```python
eval_callback = MaskableEvalCallback(
    ...
    deterministic=args.eval_deterministic,
    ...
)
eval_stats_callback = EvalStatsCallback(
    ...
    deterministic=args.eval_deterministic,
    ...
)
```

Default should be `False` (stochastic) for the next run. If you want a side-by-side
comparison, run a short diagnostic with `--eval_deterministic` to see the deterministic
baseline, then leave it off.

### 2.2 Episode Count

With stochastic eval, each episode is now a distinct sample. Increase:
```bash
--n_eval_episodes 20     # up from 10
```

20 stochastic episodes gives a reasonable estimate of mean policy reward across diverse
game states, while keeping eval time low (this env is fast).

### 2.3 Expected Behavior Change

The eval reward mean will now vary run-to-run even with an unchanged policy. This is
correct — it reflects genuine game variance. The important diagnostic is the **trend** over
training steps, not absolute values at a single checkpoint. The best_model selection by
`MaskableEvalCallback` will be based on the stochastic mean, which is noisier but more
informative.

---

## Priority 3 — Vrroomm Placement Reward Restructuring

### Motivation

The current `vrroomm_placement_bonus = +0.005` fires unconditionally whenever any Vrroomm
marker is placed, regardless of delivery viability. This signal is:
- Too weak (200× smaller than the +1.0 delivery reward)
- Uninformative (same value whether the placement will yield 3 deliveries or 0)
- Not conditioned on the game state the policy is in

Replacing it with a tiered, state-conditioned bonus gives the policy a clear signal: "I
placed a Vrroomm marker in a round where deliveries are likely/confirmed."

Additionally, the **passenger selection step** (stage 1 of Vrroomm resolution) currently
gets zero reward. The +1.0 delivery reward fires at stage 2 (destination selection). The
stage-1 step is learning from bootstrapped advantage estimates alone, with no direct signal.
A small shaping bonus at stage 1 for selecting a passenger with a valid destination
stabilizes this decision.

### 3.1 Tiered Placement Bonus

Replace `vrroomm_placement_bonus` with three config values:

```python
# In RewardConfig (config.py):
vrroomm_placement_bonus: float = 0.0          # unconditional (set to 0 — replaced by tiers)
vrroomm_bonus_confirmed: float = 0.06         # tier 1: ≥1 confirmed deliverable passenger
vrroomm_bonus_probable: float = 0.02          # tier 2: probable opportunity (indirect)
```

**Tier 1 — Confirmed opportunity:**
At the moment of Vrroomm marker placement (CHOOSING_ACTIONS), evaluate:
- `deliver_features_current = compute_delivery_features(state, player_id, current_clock_type)`
- `deliver_features_next = compute_delivery_features(state, player_id, next_clock_type)`
- `confirmed = deliver_features_current.deliverable_count > 0 OR deliver_features_next.deliverable_count > 0`

If `confirmed`: award `+vrroomm_bonus_confirmed`.

Checking BOTH current and next clock types handles the most common case where the clock
advances before Vrroomm resolves. If an opponent places a TIME_CLOCK marker and stops the
clock, the current-type check covers that scenario.

**Tier 2 — Probable opportunity:**
If not confirmed, check:
- `(has_connected_train_station AND (deliver_features_current.available_slot_count > 0 OR deliver_features_next.available_slot_count > 0))`

Where `has_connected_train_station` = any train station node is in `player_network_nodes`.

Rationale: A train station on the player's network means PASSENGERS resolution will likely
spawn passengers in an accessible location (train stations are the primary spawn points).
If delivery slots also exist on the network (current or next clock type), Vrroomm is
probably valuable even though no deliverable passengers exist right now.

If probable: award `+vrroomm_bonus_probable`.

**Tier 0 — No opportunity:**
No bonus (0.0). Placing a Vrroomm marker with no network connectivity to train stations or
passengers is pure waste.

**BUILDINGS uncertainty:** The BUILDINGS resolution also happens before Vrroomm. A building
placed on the player's network could open a delivery slot. This is captured indirectly:
if the player is placing buildings this round (has a BUILDINGS marker on the action board),
the `available_slot_count` will increase by the time Vrroomm resolves. This is NOT
pre-computed in the shaping (we don't know where buildings will go). Accept this as a known
gap — the probable-tier train-station check provides partial coverage.

### 3.2 Implementation in `reward.py`

In `RewardCalculator`:

Add a private method `_compute_vrroomm_placement_bonus(state, player_id) -> float`:

```python
def _compute_vrroomm_placement_bonus(self, state: GameState, player_id: int) -> float:
    from rl.delivery_utils import compute_delivery_features
    from core.constants import TIME_CLOCK_ORDER

    current_type = state.global_state.time_clock_position
    clock_idx = TIME_CLOCK_ORDER.index(current_type)
    next_type = TIME_CLOCK_ORDER[(clock_idx + 1) % len(TIME_CLOCK_ORDER)]

    df_current = compute_delivery_features(state, player_id, current_type)
    df_next    = compute_delivery_features(state, player_id, next_type)

    # Tier 1
    if df_current.deliverable_count > 0 or df_next.deliverable_count > 0:
        return self.config.vrroomm_bonus_confirmed

    # Tier 2 — train station proxy
    network_nodes = state.board.get_player_network_nodes(player_id)
    has_station = any(
        state.board.get_node(nid).is_train_station
        for nid in network_nodes
    )
    has_slots = df_current.available_slot_count > 0 or df_next.available_slot_count > 0
    if has_station and has_slots:
        return self.config.vrroomm_bonus_probable

    return 0.0
```

In `_compute_marker_shaping` (or wherever `vrroomm_placement_bonus` is currently applied):
replace the unconditional `self.config.vrroomm_placement_bonus` with a call to
`_compute_vrroomm_placement_bonus(state, player_id)`.

**Performance note:** `compute_delivery_features` is called twice per Vrroomm placement
step. This adds one extra BFS per call. Placement steps are ~1/149 of all steps, so the
overhead is negligible.

### 3.3 Passenger Selection Stage Shaping (+0.01)

During Vrroomm stage 1 (head `RESOLVE_VRROOMM_PASSENGER`), after the policy selects a
passenger and `_select_vrroomm_passenger_with_tiebreak` returns the actual chosen passenger,
check whether that passenger has ≥1 valid destination for the **current** clock type
(post-time-clock-resolution, so the clock type is finalized).

If valid destinations exist: inject a small bonus `+vrroomm_passenger_selection_bonus`
into the step reward.

Add to `RewardConfig`:
```python
vrroomm_passenger_selection_bonus: float = 0.01
```

**Implementation location in `bus_env.py`:**

In the `step()` method, after processing `RESOLVE_VRROOMM_PASSENGER` actions and calling
`_select_vrroomm_passenger_with_tiebreak(...)`, add the chosen passenger id to `action_info`:

```python
action_info["vrroomm_passenger_has_destination"] = len(destinations) > 0
```

where `destinations` is retrieved from `resolver.get_available_destinations(player_id, chosen_id)`.

In `reward.py`, handle the `"vrroomm_passenger_has_destination"` key in
`compute_reward_detailed` and add the bonus to `StepRewardInfo`.

### 3.4 Existing `vrroomm_placement_bonus` Value

Set `vrroomm_placement_bonus: float = 0.0` in `RewardConfig`. The tiered bonus replaces
it entirely. The old unconditional 0.005 is no longer used.

### 3.5 No Change to Waste Penalty Values

Per earlier discussion, `avoidable_waste_penalty = -0.005` and
`resolution_type1_waste_penalty = -0.03` are intentionally soft. These remain unchanged.

---

## Optional Priority 4 — Round-Weighted Delivery Incentive

### Motivation

If, after Priorities 1-3 are implemented, the policy still learns to deliver heavily in
early rounds and coasts in later rounds, a round-weighted delivery multiplier can shift the
emphasis. This is explicitly optional — Priorities 1-3 may be sufficient.

### Design

Multiply every delivery reward by a round-number factor:

```
effective_delivery_reward = delivery_reward * (1.0 + round_multiplier * round_number)
```

With `round_multiplier = 0.05`:
- Round 1: 1.05× (mild bonus)
- Round 3: 1.15×
- Round 5: 1.25×

This preserves the round-1 signal (doesn't penalize early deliveries) while making
later-round deliveries progressively more valuable.

Add to `RewardConfig`:
```python
round_delivery_multiplier: float = 0.0  # 0 = off; set to ~0.05 to enable
```

**Implementation:** In `reward.py`, wherever `self.config.delivery_reward` is multiplied
by the delivery count, multiply instead by:
```python
self.config.delivery_reward * (1.0 + self.config.round_delivery_multiplier * state.global_state.round_number)
```

**Caution:** This changes the scale of delivery rewards across rounds. The terminal reward
(point differential) is unaffected. Monitor value function residuals after enabling — if
the value function accuracy drops, lower `round_delivery_multiplier` or disable this.

**Recommendation:** Do not enable this in the same run as Priorities 1-3. Introduce it
in a subsequent run once the baseline with the observation changes is stable.

---

## Suggested Run Configuration for Next Run

```bash
python scripts/train.py \
    --use_opponent_pool \
    --multi_policy \
    --self_play_prob 0.025 \
    --sampling_method pfsp \
    --pool_size 100 \
    --pool_save_interval 10000 \
    --pool_eval_interval 25000 \
    --pool_eval_opponents 10 \
    --pool_eval_games 10 \
    --prune_strategy least_diverse \
    --total_timesteps 6000000 \
    --ent_coef 0.005 \
    --ent_coef_final 0.001 \
    --n_envs 16 \
    --n_steps 4096 \
    --batch_size 4096 \
    --target_kl 0.04 \
    --lr 7e-4 \
    --n_epochs 15 \
    --n_eval_episodes 20 \
    --randomize_training_slot \
    --disable_dist_validate \
    --diag_log_interval 100000 \
    --diag_log_samples 256 \
    --diag_log_tolerance 5e-5 \
    --skill_tracking openskill \
    --skill_temperature 30 \
    --pl_tau 30.0 \
    --openskill_recenter_interval 6000000 \
    --start_fresh_directory \
    --prime_n_top 30 \
    --prime_n_random 10 \
    --load_pool_dir logs/ppo_bus_20260502_112313/opponent_pool \
    --use-slot-actionability \
    --use-delivery-features          # NEW: P1
    # --eval_deterministic NOT set  # NEW: P2 (stochastic eval by default)
```

**What changes from the last run:**
- `--use-delivery-features` (P1): +33 obs dims, obs goes from 1512 → 1545
- No `--eval_deterministic` (P2): stochastic eval, `n_eval_episodes 20`
- No `--initial_checkpoint` (fresh start due to obs dim change)
- Reward changes (P3): tiered Vrroomm placement bonus + passenger selection shaping
  — these require code changes, not CLI flags

**Priming phase (bootstrap run, ~500K steps first):**
Run a short bootstrap to populate the pool before the full run. Or prime directly from
the 20260502 pool as shown above — the priming copies existing checkpoints but the new
policy starts from random init, so early games will be uncompetitive. The pool diversity
will accelerate early learning. If early performance is poor (reward < 5.0 at 500K),
consider running a shorter pure-selfplay warmup first.

---

## File Change Summary

| File | Change |
|------|--------|
| `rl/delivery_utils.py` | **NEW** — `DeliveryFeatures` dataclass + `compute_delivery_features()` |
| `rl/config.py` | Add `use_delivery_features` flag; add `passenger_feature_dim` and updated `global_features_size` properties; add `vrroomm_bonus_confirmed`, `vrroomm_bonus_probable`, `vrroomm_passenger_selection_bonus` to `RewardConfig`; set `vrroomm_placement_bonus = 0.0` |
| `rl/observation.py` | Update `_encode_passengers` signature; add delivery features (2/passenger + 3 global) when flag enabled; use `passenger_feature_dim` property instead of ClassVar |
| `rl/reward.py` | Replace unconditional placement bonus with `_compute_vrroomm_placement_bonus()`; handle `vrroomm_passenger_has_destination` in `compute_reward_detailed`; add to `StepRewardInfo` |
| `rl/bus_env.py` | After `_select_vrroomm_passenger_with_tiebreak`, set `action_info["vrroomm_passenger_has_destination"]` |
| `scripts/train.py` | Add `--use-delivery-features` and `--eval_deterministic` CLI flags; pass to callbacks/config |
| `bash_scripts/pfsp_ppo_v2.bash` | **NEW** — Two-phase training script (priming + pool play) with all new flags |

---

## Implementation Status

**Implemented:** 2026-05-15

### Priority 1 — Observation Enrichment ✅
- Created `rl/delivery_utils.py`: `DeliveryFeatures` dataclass and `compute_delivery_features()` function. Single BFS call per (player, building_type) pair; replicates VrrooommResolver slot-occupancy logic without coupling to the resolver.
- `rl/config.py`: Added `use_delivery_features: bool = False` to `ObservationConfig`; added `passenger_feature_dim` property (5→7 when enabled); updated `passenger_features_size` to use the property; updated `global_features_size` to add 3 when enabled.
- `rl/observation.py`: Added `compute_delivery_features` import; updated `_encode_passengers` signature to accept `current_player_id`; added 2 per-passenger features (`is_reachable_by_current_player`, `is_valid_delivery_source`) when flag enabled; added 3 global features (`my_deliverable_count_current`, `my_deliverable_count_next_clock`, `my_available_slots_current`) at the tail of `_encode_global`.
- Obs dim verified: 1470 → 1503 (delivery only) → 1545 (both flags). All values confirmed by unit test.

### Priority 2 — Stochastic Evaluation ✅
- `scripts/train.py`: Added `--eval_deterministic` flag (`store_true`; default False = stochastic). Both `MaskableEvalCallback` and `EvalStatsCallback` now receive `deterministic=args.eval_deterministic` instead of hardcoded `True`.
- `n_eval_episodes` default already set to 20 in train.py.

### Priority 3 — Vrroomm Reward Restructuring ✅
- `rl/config.py` (`RewardConfig`): `vrroomm_placement_bonus` set to `0.0`; added `vrroomm_bonus_confirmed = 0.06`, `vrroomm_bonus_probable = 0.02`, `vrroomm_passenger_selection_bonus = 0.01`.
- `rl/reward.py`: Imported `compute_delivery_features` and `TIME_CLOCK_ORDER` at top. Added `_compute_vrroomm_placement_bonus(state, player_id)` method implementing Tier 1/2/0 logic. Updated `_compute_marker_shaping` to accept `player_id` and call the tiered method for Vrroomm. Added `vrroomm_passenger_selection_bonus` field to `StepRewardInfo` (included in `total`). Added handling for `vrroomm_passenger_has_destination` key in `compute_reward_detailed`.
- `rl/bus_env.py`: After `_select_vrroomm_passenger_with_tiebreak`, captures `chosen_id` return value; filters `valid_actions` for that passenger's destinations; sets `action_info["vrroomm_passenger_has_destination"]`.

### Priority 4 (Optional) — Not implemented
- Deferred pending results from Priorities 1-3. See design in document above if needed.

### Bash Script
- `bash_scripts/pfsp_ppo_v2.bash` (new file): Two-phase training script.
  - Phase 1 (uncommented): Pure self-play priming from scratch with both obs flags. 5M steps, lr=5e-4, ent_coef 0.05→0.01, n_steps=4096.
  - Phase 2 (commented): PFSP pool play loading from Phase 1. 6M steps, lr=7e-4, target_kl=0.04, n_epochs=15, stochastic eval, n_eval_episodes=20. Update `PRIMING_RUN` placeholder with actual Phase 1 run name before running.
  - Note: Old pool checkpoints (ppo_bus_20260502_112313, obs dim 1512) are **incompatible** with the new obs dim (1545). Phase 1 must start from scratch with no pool loading.

---

## Implementation Summary

All three priorities from this document are implemented and smoke-tested. The key changes are:

1. **The policy now directly observes which passengers it can pick up and deliver** — no need to implicitly infer rail connectivity from 70 edge features. Two binary per-passenger flags and three global scalars give first-order delivery information. This should most directly address the mid-to-late game delivery failure.

2. **Eval now measures genuine policy variance** — stochastic sampling means each eval episode can follow a different trajectory, making the eval curve meaningful instead of N=1 noise.

3. **Vrroomm placement signal is clear and state-conditioned** — the old flat +0.005 is replaced by +0.06 (confirmed deliverable) or +0.02 (probable via station+slots), with +0.01 additional shaping for selecting a passenger that has a valid destination. The scale is now meaningful relative to the +1.0 delivery reward.

The obs dim change (1512→1545) requires a fresh priming run. `bash_scripts/pfsp_ppo_v2.bash` Phase 1 bootstraps this from scratch.
