# Hierarchical PPO Plan for BusRL

**Status:** Draft for implementation
**Goal:** Replace the flat 1,670-action MaskablePPO policy with a phase-aware hierarchical PPO policy that is trainable and stable, using a shared encoder and multiple heads. This plan is scoped to be actually trainable with SB3 + MaskablePPO.

## 0. Scope and Priorities (Locked)

- **Primary priority:** stability and clean integration with the current SB3 + MaskablePPO stack.
- **Board scope:** hard-locked to the default board topology (no dynamic board support in this phase).
- **Line expansion scope:** only edges present in the board graph are represented; legality is enforced by existing game rules via masks.
- **Vrroomm scope:** passengers at the same origin may be treated as identical for action selection.
- **MCTS scope:** explicitly lower priority; can lag until hierarchical PPO is stable.

---

## 1. Summary of the Approach

We will implement a **hierarchical action interface** first, then layer policy complexity only if needed.

### Phase 1 (stability-first)

- **Shared encoder** (same feature extractor / MLP used today)
- **Single policy output head** with `Discrete(max_head_size)`
- **Context-dependent masks** that activate only the current logical head region
- **Step-by-step RESOLVING_ACTIONS decisions** through persistent resolver state

### Phase 2 (optional later)

- True multi-head policy module if Phase 1 bottlenecks are observed

In both phases, the decision structure is hierarchical by context, but we prioritize minimal disruption to the current framework.

Core mechanics:

- **Shared encoder** (same feature extractor / MLP used today)
- **Logical action heads** (one per game phase or resolver area), implemented via masks first
- **Single action space**: `Discrete(max_head_size)`
- **Per-head masking** to ensure only valid actions are sampled

We will move **RESOLVING_ACTIONS** into a **step-by-step** RL decision process (instead of fully auto-resolving), using the existing `ActionResolver` to determine valid choices at each point.

---

## 2. Decision Contexts (Heads)

Each decision point maps to a **head ID**, derived from the game phase and, during resolution, the current action area.

**Heads (proposed):**

### Setup Phase
- `SETUP_BUILDINGS` head
- `SETUP_RAILS_FORWARD` head
- `SETUP_RAILS_REVERSE` head

### Choosing Phase
- `CHOOSING_ACTIONS` head

### Resolving Phase (per action area)
- `RESOLVE_LINE_EXPANSION` head
- `RESOLVE_PASSENGERS` head
- `RESOLVE_BUILDINGS` head
- `RESOLVE_TIME_CLOCK` head
- `RESOLVE_VRROOMM_PASSENGER` head (new subphase)
- `RESOLVE_VRROOMM_DEST` head (new subphase)

Automatic areas (no decision):
- `BUSES`, `STARTING_PLAYER` are always resolved automatically by the resolver.

---

## 3. Head Sizes (Concrete and Trainable)

We will use **fixed head sizes**, but will only mask valid actions at runtime.

### 3.1 Setup Heads
| Head | Size | Notes |
|------|------|------|
| Setup Buildings | `36 nodes * 2 slots * 3 types = 216` | One action per placement |
| Setup Rails (FWD/REV) | `70 edges` | One edge per action |

### 3.2 Choosing Head
| Head | Size | Notes |
|------|------|------|
| Choosing Actions | `7 action areas + PASS = 8` | Slot index is implicit |

### 3.3 Resolving Heads
| Head | Size | Notes |
|------|------|------|
| Line Expansion | `140` | Enumerate all (edge, endpoint) pairs where endpoint is one end of that edge |
| Passengers | `6 distributions` | Same as current action space |
| Buildings | `36 nodes * 2 slots * 3 types = 216` | One action per placement |
| Time Clock | `2` | advance or stop |

### 3.4 Vrroomm (Two-Stage Reduction)
We split Vrroomm into two sub-steps:

**Stage 1: Pick Passenger**
- Head size: `MAX_PASSENGERS + SKIP`
- `15 + 1 = 16`
- Implementation note: passengers at the same origin can be treated as equivalent in policy selection.
  - Deterministic tie-break: if the policy selects an origin-grouped passenger, map to the lowest `passenger_id` at that origin for resolver input.

**Stage 2: Pick Destination Slot**
- Head size: **All building slots on the board**
- `47 slots` (derived from current default board)

This is much smaller than the original 1081 size.

**Final max head size:**
- `max(216, 140, 47, 16, 8, 70, 2)` → **216**

So the new `action_space = Discrete(216)`.

Observation head/context additions (summary):
- `head_id` one-hot: length 10 (one per head; no reserved slots)
- `vrroomm_stage` one-hot: length 2 (`[stage1, stage2]`), all-zeros outside Vrroomm

---

## 4. Hierarchical Action Mapping

### 4.1 New Mapping Logic
Create a new mapping utility (new file or extend `rl/action_space.py`) to provide:

```python
head_id = get_head_id(state, resolver_context)
valid_actions = get_valid_actions_for_head(state, resolver_context)
mask = build_mask_from_valid_actions(head_id, valid_actions)
```

Each head will define:
- **deterministic ordering** of its action list
- **index ↔ action mapping** within head range

Integration-first rule:
- Keep a single global `Discrete(216)` output in the first implementation.
- Treat heads as masked index sets within that shared output (indices `0..head_size-1` reused per head).
- Avoid a custom multi-output policy class until this is stable.

### 4.2 Perfect Masking Guarantees
- Build masks only from engine/resolver valid actions; no heuristic additions.
- For each head, valid indices are exactly `0..(len(valid_actions)-1)` mapped to the deterministic action list.
- Mask must be all-false outside the current head's index set.
- If mask is empty during RESOLVING_ACTIONS, call `resolver.advance()` until `AWAITING_INPUT` or `ALL_COMPLETE` (do not query the policy).
- If mask is empty in a decision phase (setup/choosing), raise with phase + resolver context to surface a bug.
- Add runtime assertions: `mask.any()` in decision phases, no NaNs, and selected index maps to a valid action.

### 4.3 Deterministic Ordering Examples
- Setup buildings: sort by `(node_id, slot_index, building_type)`
- Line expansion: sort by `(edge_id, from_endpoint)` using only endpoints belonging to each edge in the fixed graph
- Buildings: same as setup
- Vrroomm passenger: sort by `passenger_id`
- Vrroomm destination: map `(node_id, slot_index)` to a fixed slot index via board topology order

---

## 5. BusEnv / Resolver Integration

### 5.0 Engine Constraint (Current Code)

Current `GameEngine.step()` does not execute resolution-phase action types directly for RL, and `_get_valid_resolution_actions()` returns empty. Therefore, hierarchical resolving decisions must be driven through a persistent `ActionResolver` inside `BusEnv` during this implementation phase.

### 5.1 Step-by-Step Resolution
Replace the current `_auto_resolve_actions()` (which fully resolves) with a **step-based process**:

- On entering `RESOLVING_ACTIONS`, create a persistent `ActionResolver` and call `start_resolution()`.
- While resolver context is not `AWAITING_INPUT`, call `advance()` automatically.
- When `AWAITING_INPUT`, build the head + action list from resolver context.
- Apply chosen resolution actions by calling resolver APIs directly (`apply_action` / `skip_vrroomm_deliveries`), not by routing through `GameEngine.step()`.

### 5.2 Syncing GameState
When resolver advances areas or slots, explicitly update:

- `state.global_state.current_resolution_area_idx`
- `state.global_state.current_resolution_slot_idx`

This is required so `PhaseMachine.should_end_resolving_phase()` can advance to CLEANUP reliably.

Implementation detail:
- Add a small synchronization helper in `BusEnv` that mirrors resolver context into `GlobalState` after every resolver transition.
- Prefer exposing a lightweight accessor/helper on `ActionResolver` for canonical `(area_idx, slot_idx)` values instead of duplicating index logic in multiple places.

### 5.3 Auto-Advance Reliability
We must ensure **no deadlocks** where the environment gets stuck with no valid actions.

**Plan:**
- Add diagnostic logging in BusEnv for any empty-valid-actions in decision phases.
- During RESOLVING_ACTIONS, empty action sets for a marker are legitimate; the resolver should auto-advance without requesting policy input.
- Verify resolver paths in these cases:
  - player cannot place rails in Line Expansion
  - no passengers can be delivered
  - no building slots are valid
- Ensure the resolver's `_advance_to_next_area_with_markers()` always reaches `ALL_COMPLETE` when appropriate.

---

## 6. Hierarchical Policy

Create `rl/hierarchical_policy.py` implementing:

```python
class HierarchicalMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    # shared encoder
    # dict of action heads
    # selects head from observation
```

Stability-first adjustment:
- First implementation can remain on `MaskableActorCriticPolicy` with shared `Discrete(216)` output and head-aware masks.
- Introduce `HierarchicalMaskableActorCriticPolicy` only if needed after baseline training is stable.

### Logit Stability (Optional)
- Add optional logit clamping after masking: default range `[-20, 20]` via `torch.clamp`.
- Keep this behind a training flag to compare stability vs baseline masking behavior; also allow `torch.nan_to_num` if needed.

### Head Selection
- Use observation global features:
  - phase one-hot
  - resolution area one-hot
- For Vrroomm two-stage, include a small internal flag in the env (e.g., `env._vrroomm_stage`) and expose it as a **one-hot** feature (defaults to all-zeros outside Vrroomm resolution).
- This requires a small observation shape update in `ObservationConfig` and encoder logic.
- Add an explicit **head_id one-hot vector** in the observation to disambiguate heads cleanly (recommended).
  - Size = number of heads (currently 10); no reserved slots.
- Vrroomm stage must be included (phase/area alone does not disambiguate stage 1 vs stage 2).
  - Use a 2-length one-hot `[stage1, stage2]`, and all-zeros outside Vrroomm.

---

## 7. Training Script Changes

- Add `--hierarchical` flag in `scripts/train.py`
- Add `--logit-clamp` flag (default clamp range `[-20, 20]`) to enable optional logit clamping after masking
- Default to hierarchical policy class if enabled
- Update evaluation to load the custom policy class

---

## 8. MCTS Compatibility

`rl/mcts.py` currently calls `policy.action_net(latent)` directly.
This will break with multi-head.

**Plan priority:** defer MCTS migration until hierarchical PPO training is stable.

When re-enabled, **fix** by using policy distribution APIs (with masks) rather than direct `action_net` assumptions.

---

## 9. Test Plan

### Unit Tests
- Verify each head size and action list size
- Ensure mapping is deterministic and reversible per head
- Verify default-board locked constants (`36 nodes`, `70 edges`, `47 slots`, `15 passengers`) are consistent with mapping tables

### Integration Tests
- Full game run with hierarchical env
- Verify resolver advances across all resolution areas
- Ensure no step yields empty valid action list in decision phases; in RESOLVING_ACTIONS, empty-action slots auto-advance without policy input
- Verify resolver-to-`GlobalState` synchronization after every resolver transition
- Verify line-expansion actions are only generated from valid board edges/endpoints
- Verify Vrroomm stage-1 equivalence behavior for passengers sharing an origin

### Training Smoke Test
- MaskablePPO trains for at least 5k steps without crash
- No NaNs in policy output
- Run smoke on the integration-first (single-head masked) architecture before any custom multi-head policy class

---

## 10. Follow-Up Investigation Notes

1. **ActionResolver auto-advance**
- Ensure the resolver's `_advance_to_next_area_with_markers()` always reaches `ALL_COMPLETE` when appropriate.
   - Add a test case for each resolver where zero actions exist.

2. **Deadlock Prevention**
   - If env sees `len(valid_actions) == 0` in a decision phase, raise clearly and dump resolver context.
   - This will catch any silent failures early in training.

---

## 11. Final Notes

This plan is designed so we can **actually train** in SB3:

- `action_space` shrinks from 1670 → 216
- Action masks are lightweight and stable
- Resolver-driven decision points are explicit and testable

Implementation order:
1. New mapping + resolver-stepped `BusEnv` with single shared policy output and head-aware masks.
2. Stabilize training and tests.
3. Optional: custom multi-head policy.
4. Optional: MCTS compatibility update.
