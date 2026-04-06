# Hierarchical RL Implementation Checklist (BusRL)

**Purpose:** Concrete, file-by-file checklist aligned to `hierarchical_rl_plan.md` for implementing the hierarchical action interface and training-ready PPO setup.

---

## 0. Baseline & Guardrails
- [ ] **Replace legacy flat 1670 mode** (no dual-mode support).
- [ ] Add/confirm a single switch for hierarchical mode (e.g., `BusEnv(action_mode="hierarchical")` or `--hierarchical` in scripts).
- [ ] Add runtime assertions for **perfect masking** (no NaNs, mask non-empty in decision phases, action index maps to a valid action).

---

## 1. Action Mapping Layer (New or Extension)
**Files:** `rl/action_space.py` (extend) or new `rl/hierarchical_action_space.py`
- [x] Added `rl/hierarchical_action_space.py` (new mapping module).
- [x] Added tests in `tests/test_hierarchical_action_mapping.py`.

- [x] Define **Head IDs** for all decision contexts (10 total):
  - Setup: `SETUP_BUILDINGS`, `SETUP_RAILS_FORWARD`, `SETUP_RAILS_REVERSE`
  - Choosing: `CHOOSING_ACTIONS`
  - Resolving: `LINE_EXPANSION`, `PASSENGERS`, `BUILDINGS`, `TIME_CLOCK`, `VRROOMM_PASSENGER`, `VRROOMM_DEST`
- [x] Implement `get_head_id(state, resolver_ctx, vrroomm_stage)`.
- [x] Implement **deterministic ordered action lists** for each head (seed-independent):
  - Note: building action catalog uses fixed 36*2*3=216 index space for compatibility.
    Some (node, slot) pairs may not exist on the board and will be masked out by valid-action generation.
  - Ensure `CHOOSING_ACTIONS` head maps to **action areas**, not slots (slots assigned A/B/C automatically).
  - Setup buildings: `(node_id, slot_idx, building_type)`
  - Setup rails: `edge_id`
  - Choosing: action areas + pass
  - Line expansion: `(edge_id, from_endpoint)` pairs (2 per edge)
  - Passengers: 6 distributions
  - Buildings: `(node_id, slot_idx, building_type)`
  - Time clock: `[advance, stop]`
  - Vrroomm passenger: passenger IDs (with equivalence handling)
  - Vrroomm dest: **47 building slots** ordered by board topology
- [x] Implement **head-local index <-> action mapping**; indices are reused per head (0..head_size-1).

---

## 2. Vrroomm Two-Stage Handling
**Files:** `rl/bus_env.py`, mapping module, `engine/action_resolver.py` (if helper is added)

- [x] Enforce Vrroomm stage transitions: 0 -> 1 on enter Vrroomm, 1 -> 2 after passenger chosen, 2 -> 0 after slot resolved or skip. (helper state machine)
- [x] Introduce env-level `vrroomm_stage` state: 0/1/2 (0 = not in Vrroomm, 1 = passenger, 2 = destination).
- [x] **Stage 1:** head selects passenger (plus `SKIP`). (stage state supports selection)
- [x] **Stage 2:** head selects destination building slot (index into 47-slot list). (stage state builds delivery action)
- [x] Deterministic tie-break for passenger equivalence: if grouped by origin, choose the lowest `passenger_id` at that origin. (implemented in BusEnv)
- [x] Ensure `skip_vrroomm_deliveries()` is called when `SKIP` chosen. (skip helper added; resolver wiring pending)

---

## 3. BusEnv Resolver Integration (Step-by-Step)
**Files:** `rl/bus_env.py`

- [x] Assert chosen action index maps to a valid action in the current head's action list.
- [x] Ensure Vrroomm passenger selection is restricted to resolver-valid passengers; assert chosen passenger is in valid list.
- [x] Ensure line expansion choices are restricted to resolver-valid placements; assert chosen (edge, from_endpoint) appears in valid actions.
- [x] Replace `_auto_resolve_actions()` full resolution with **persistent `ActionResolver`** usage.
- [x] On entering `RESOLVING_ACTIONS`, create resolver and `start_resolution()`.
- [x] While resolver status is not `AWAITING_INPUT`, call `advance()` automatically.
- [x] If resolver returns **no valid actions** for a slot, auto-advance (no policy call).
- [x] Ensure `action_masks()` uses resolver context for valid actions when awaiting input.

---

## 4. Resolver <-> GlobalState Sync Helper
**Files:** `rl/bus_env.py` (and optionally `engine/action_resolver.py`)

- [x] Add a small helper (e.g., `_sync_resolution_state(resolver_ctx)`):
  - Updates `state.global_state.current_resolution_area_idx`
  - Updates `state.global_state.current_resolution_slot_idx`
- [x] Call sync **after every resolver transition** (start, advance, apply_action, finalize).
- [x] Prefer exposing `ActionResolver.get_context()` and map its `current_area_idx` / `current_slot_idx` directly.

---

## 5. Perfect Masking Guarantees
**Files:** `rl/bus_env.py`, mapping module

- [x] Masks are built **only** from valid actions returned by engine/resolver.
- [x] Outside head range: mask must be all-false.
- [x] Assert `mask.sum() <= head_size` to prevent leakage.
- [x] Implement `build_mask(head_id, valid_actions)` that maps only valid indices for that head.
- [x] If mask empty in **setup/choosing**, raise error with phase + resolver context.
- [x] If mask empty in **resolving**, auto-advance.
- [x] Add runtime checks: `mask.any()` in decision phases, no NaNs, and chosen index maps to a valid action.

---

## 6. Observation Updates (Head & Vrroomm Stage)
**Files:** `rl/config.py`, `rl/observation.py`

- [x] Add **head_id one-hot** length 10 to observation (append at end of flat vector).
- [x] Add **vrroomm_stage one-hot** length 2 (append at end; all zeros outside Vrroomm).
- [x] Update `ObservationConfig.GLOBAL_FEATURE_DIM` (+12 total) and `total_observation_dim`.
- [x] Add encoding logic in `_encode_global()` or a new dedicated block.
- [x] Ensure observation order is documented so head features are stable.

---

## 7. Action Space Size & Env Mode
**Files:** `rl/bus_env.py`, `rl/config.py`

- [x] Set `action_space = Discrete(216)` (legacy 1670 removed).
- [x] Ensure `action_masks()` returns the correct size in all phases.

---

## 8. Logit Clamping (Optional Stability)
**Files:** `scripts/train.py`, policy module (or wrapper)

- [x] Assert value predictions are finite (fail fast if NaN/inf).
- [x] Add `--logit-clamp` flag (default `False`).
- [x] When enabled, clamp masked logits to `[-20, 20]` **right before distribution creation**.
- [x] Add `torch.isfinite` assertion on logits after masking/clamp (fail fast on NaN/inf).
- [ ] If needed, optionally apply `torch.nan_to_num`.

---

## 9. Policy Class (Optional Multi-Head)
**Files:** `rl/hierarchical_policy.py`, `rl/__init__.py`

- [x] Only implement if the shared-output + mask approach is insufficient.
- [ ] If implemented, use shared encoder + per-head linear layers.
- [x] Ensure SB3 serialization imports class from `rl.__init__`.

---

## 10. Training / Eval Script Changes
**Files:** `scripts/train.py`, `scripts/evaluate.py`

- [x] Track head usage counts during rollouts and log to TensorBoard (ensure all heads are exercised).
- [ ] Add `--hierarchical` flag (enables hierarchical env + action size).
- [x] Add `--logit-clamp` flag to enable optional clamping.
- [x] Ensure model loading works when custom policy is used (import path available).

---

## 11. MCTS Compatibility
**Files:** `rl/mcts.py`

- [ ] Defer until PPO is stable.
- [ ] When re-enabled: use policy distribution APIs with action masks, not direct `action_net` logits.

---

## 12. Tests & Diagnostics
**Files:** `tests/` (new or extend existing)

- [ ] Unit tests for each head mapping size and deterministic ordering.
- [x] Unit test: BusEnv resolver decision flow for Time Clock.
- [x] Unit test: observation tail includes head_id + vrroomm_stage one-hots.
- [ ] Integration test: full game with hierarchical env completes without deadlock.
- [ ] Test: empty valid actions during resolving **auto-advance** without policy call.
- [ ] Test: `head_id` and `vrroomm_stage` observation bits correct across phases.
- [ ] Smoke test: MaskablePPO trains 5k steps without NaNs.

---

## Decisions Locked In
- Legacy flat 1670 mode: **removed** (hierarchical only).
- head_id + vrroomm_stage: **append at end** of flat observation vector.
- Logit clamping: **right before distribution creation**, with NaN/inf assertion.

---

## Suggested Additions (Optional)
- Add a debug info field in `info` dict: `current_head_id`, `vrroomm_stage`, and `valid_action_count` per head.
- Add a simple `env.validate_mask()` helper to centralize mask sanity checks.
- Add a `--hierarchical-debug` flag to log resolver context when masks are empty.

---

## Notes (Revisit Later)
- Observation reads currently call `_get_decision_context()` to compute `head_id`, which can auto-advance the resolver during `RESOLVING_ACTIONS`. This keeps observations aligned with real decision points, but means “read-only” observation calls can mutate state. We’re leaving this behavior in place for now; revisit if we need side-effect-free observation reads.
