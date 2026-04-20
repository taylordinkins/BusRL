# Wasted Marker Analysis & Proposals

This document captures the diagnosis of the persistent wasted-marker problem observed across PFSP training runs, and concrete proposals for addressing it. Written after analysis of run `ppo_bus_20260412_104509`.

---

## Background

Three action areas — `line_expansion`, `passengers`, and `buildings` — use M#oB (max buses owned by any player) to determine how many slots actually resolve each round. Markers placed in slots beyond M#oB are guaranteed to waste. Markers placed within M#oB slots can still waste if no valid resolution target exists (no valid path, no available passenger, no buildable building).

The `eval/waste_avoidable_rate_est` metric tracks the latter case and has been **stuck at 0.875–0.95 across the entire run with no improvement trend**, indicating the current reward signal is not moving the needle on this behavior.

---

## 1. Two Distinct Waste Types

Understanding which type of waste is occurring is prerequisite to fixing it.

**Type 1 — M#oB slot waste:** The agent places in a slot where `slot_index >= max_buses`. That slot can never execute regardless of board state. Example: placing in slot C(2) when M#oB=2.

**Type 2 — Avoidable actionable waste:** The agent places in a slot within M#oB, but at resolution time no valid action exists in that area (no valid line to expand, no available passenger, no buildable building). The slot could execute but nothing to execute against.

These require different fixes. The current reward shaping in `reward.py:_compute_marker_shaping()` only addresses Type 1, and only at placement time.

---

## 2. Current Signal Is Too Weak

**Current behavior:** `reward.py:175–207` applies ±0.02 at marker placement time:
- `+marker_opportunity_bonus (0.02)` if the slot is actionable given projected M#oB
- `-avoidable_waste_penalty (0.02)` if the slot is projected to waste

**Why it isn't working:**

1. **Magnitude.** ±0.02 against game rewards of 2–10 is 1–2% of the reward scale. In a 4-player game where a single player's marginal contribution to the episode reward is roughly 25%, this signal is below the noise floor and will be swamped by variance in opponent behavior and game outcomes.

2. **No signal for Type 2.** `avoidable_est` and `waste_avoidable_rate_est` are pure metrics — no gradient flows from them. The policy receives zero feedback when a within-M#oB marker resolves to nothing.

3. **Credit assignment lag.** Placement occurs several turns before resolution. Even if the placement-time signal is correct, the policy must bridge a multi-step gap between the action and any outcome. The current magnitude cannot sustain this over the policy's effective horizon.

**Proposals:**

- Raise `avoidable_waste_penalty` to **-0.1** and `marker_opportunity_bonus` to **+0.05** (asymmetric is intentional — punishing waste harder than rewarding good placement). These values are roughly 5–10× the current magnitude and sit in a range where the signal should be distinguishable from reward noise.
- Consider a separate, higher penalty specifically for Type 1 waste (guaranteed by M#oB), since that case is unambiguously bad and fully observable at placement time.

---

## 3. No Resolution-Time Feedback (Highest Impact Proposal)

**Current behavior:** `bus_env.py:457–499` (`_maybe_record_resolution_waste()`) detects at resolution time that a marker wasted, populates `_resolution_waste_by_area`, and the callback logs it. No reward is emitted.

**Why it matters:** The policy never receives a learning signal at the moment the waste actually occurs. This is the tightest possible credit assignment — the penalty fires at the exact transition where the cost is realized — and it is currently unused.

**Proposal:** Wire `_maybe_record_resolution_waste()` (or a new parallel method) to emit a shaped reward at resolution time. Suggested values:

- **-0.1 per wasted marker** at the resolution transition, applied to the player who placed it
- Applied only to Type 1 waste (M#oB) to avoid double-penalizing with any future Type 2 signal, or applied to both with differentiated magnitudes

The infrastructure already computes exactly who placed which marker and whether it wasted (`slot.player_id`, `slot_index >= max_buses`). This is the lowest-implementation-cost, highest-signal-quality change available.

---

## 4. Missing Per-Slot Actionability Feature in Observation

**Current behavior:** `observation.py:323–388` (`_encode_action_board()`) encodes 4 features per slot: `is_occupied`, `occupying_player_relative_idx`, `is_current_player`, `placement_order`. M#oB is encoded separately in global features (`observation.py:502`). The agent must infer "is this slot actionable given current M#oB" by computing `(M#oB_global - slot_index) > 0` from these two separate feature groups.

**Why it matters:** This is a learnable computation, but it requires the policy network to cross-reference global and local features and maintain the conjunction across layers. Making it explicit removes a reasoning step and gives the value function a direct handle on placement quality.

**Proposal:** For each slot in `{line_expansion, passengers, buildings}`, add a single binary feature to the slot encoding:

```
is_actionable = float((projected_max_buses - slot_index) > 0)
```

This uses the same projection logic already in `reward.py:_projected_max_buses_for_area()`. Adding it to the observation makes the same information available to both the policy and value heads without any new computation — it's purely a feature engineering change.

**Caveat:** This increases the observation dimension by up to 18 features (3 areas × 6 slots). Any checkpoint loaded from a prior run will be incompatible unless the observation config is versioned and the new features are zero-padded on load.

---

## 5. Verify the Metric Before Tuning Further

**Current concern:** `eval/waste_avoidable_rate_est` is computed as:

```
avoidable_est = max(0, actionable - opportunities)
avoidable_rate = avoidable_est / actionable
```

where `opportunities` counts resolution slots where `valid_action_count > 0`. A rate of 0.875–0.95 means the agent is placing markers in areas where valid resolution actions exist less than 5–12.5% of the time. This is either a genuine game constraint (the board genuinely has no valid targets most rounds) or a counting artifact in the callback.

**Proposal:** Before committing to reward or observation changes, add episodic debug logging for a small number of eval games to confirm the rate reflects real behavior. Specifically: log the area, slot, M#oB, and `valid_action_count` at resolution time for every placed marker in 10–20 eval episodes. If the rate is real, that informs the urgency of the resolution-time signal (Proposal 3). If it is a metric artifact, fix the metric first to avoid tuning against a misleading number.

---

## Priority Order

| Priority | Proposal | Effort | Expected Impact |
|----------|----------|--------|----------------|
| 1 | Resolution-time waste penalty (Proposal 3) | Low — infrastructure exists | High — closes credit assignment gap |
| 2 | Verify avoidable_rate metric is accurate (Proposal 5) | Low — logging only | High — validates or invalidates the metric |
| 3 | Raise penalty/bonus magnitudes (Proposal 2) | Trivial — config change | Medium — helps Type 1 signal only |
| 4 | Per-slot actionability feature (Proposal 4) | Medium — obs dimension change, breaks checkpoints | Medium — reduces inference burden on policy |

Proposals 1 and 3 can be done together in the same code change without risk. Proposal 4 should be deferred until a checkpoint boundary (start of a new primed run) to avoid observation mismatch.

---

This document is intended as a reference for the next implementation phase. Update with findings from Proposal 5 (metric verification) before committing to specific magnitude values.
