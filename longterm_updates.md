# Long-Term Updates (Living Document)

This document captures longer-horizon follow-ups that we may want to revisit once the hierarchical PPO baseline is stable. It is intentionally non-checklist and can evolve as we learn from training runs.

---

## 1. Observation Side-Effects During Resolver Auto-Advance

**Current behavior:** `BusEnv._get_observation()` calls `_get_decision_context()`. During `RESOLVING_ACTIONS`, this can auto-advance the `ActionResolver` until an actionable decision is reached. This makes observations line up with real decision points, which is good for policy training, but it introduces **side effects** when an observation is queried “just to inspect state.”

**Why it matters:**
- In debugging tools, tests, or alternate inference paths, a call to get the observation can mutate resolver state and advance phases.
- This is acceptable for training (policy sees only actionable states), but it can surprise downstream tooling if someone expects observations to be read-only.

**Possible future change:**
- Add a “read-only observation mode” where `_get_observation()` computes `head_id` without advancing the resolver.
- Keep the auto-advance path inside `action_masks()` and `step()` only.
- Optional: expose a helper `peek_decision_context()` that reads resolver context without advancing.

**Decision for now:** keep current behavior; revisit if debugging or evaluation workflows require strict read-only observations.

---

## 2. OpenSkill Win Metric (Tie Handling)

**Current behavior (OpenSkill eval):**
- Win is defined as `current_score >= max_score`.
- Ties are treated as wins, so early runs (identical policies) show `win_rate=1.0`.

**Why it matters:**
- This inflates “win rate” when the pool is small and opponents are similar.
- Makes it harder to interpret early learning progress.

**Potential improvements:**
1. **Ties = 0.5**  
   - More standard for head-to-head metrics.
   - Keeps win rate meaningful when matches are close.
2. **Track win/tie/loss separately**  
   - Log `win_rate`, `tie_rate`, `loss_rate`.
   - Gives a clearer picture of progress as the pool diversifies.
3. **Skip initial eval if pool size is 1**  
   - Avoids a misleading “perfect” win rate at step 0.

**Decision for now:** keep as-is to minimize changes; revisit once the pool has meaningful diversity.

---

## 3. MCTS Compatibility with Hierarchical Policy

**Current status:**
- MCTS remains deferred in the checklist.
- MCTS utilities still assume direct access to logits from the policy.
- With hierarchical masking and the 216 shared action space, this can produce incorrect priors if not integrated carefully.

**What needs to change:**
1. **Use policy distribution APIs, not raw `action_net` logits**
   - MCTS should call `policy.get_distribution(obs, action_masks=mask)` to obtain masked logits.
   - This guarantees invalid actions are masked, and aligns with MaskablePPO.
2. **Action masks must be supplied for every MCTS rollout step**
   - MCTS needs to query `env.action_masks()` at each node.
   - If the observation is advanced to a decision point, masks remain consistent.
3. **Head context alignment**
   - If observations embed `head_id` + `vrroomm_stage`, ensure MCTS uses the same observation interface as training.
   - If we later make observations read-only, MCTS should call the “advance to decision point” helper explicitly.
4. **Value head sanity**
   - Ensure the custom policy’s value predictions are finite during tree search.
   - Keep logit clamping consistent with training (default on).

**Suggested MCTS integration plan (high-level):**
- Replace direct `policy.action_net(latent)` calls with `policy.get_distribution(...)`.
- Provide masks from the environment on every node expansion.
- Align policy evaluation with the same `BusEnv` wrapper used in training.

**Decision for now:** keep MCTS paused until PPO baseline stabilizes.

---

## 4. General Tracking Notes

These items aren’t blockers but can increase clarity and debug-ability:
- Log `head_id` and `vrroomm_stage` in `info` (already done).
- Head usage now logs labeled keys (e.g., `h3_choose`) and prints a legend at training start.
- Track tie rate explicitly once OpenSkill evaluation is adjusted.
- MultiPolicy fallback still references legacy NOOP index for rare empty-mask cases. It should never trigger under the new resolver flow, but if it does, it would be an invalid index with the 216 action space. Consider replacing with a safe valid-action fallback if it ever appears in logs.

---

This document is meant to evolve. If new training behaviors or stability issues crop up, add notes here with the same “why it matters / possible change / current decision” format so we can prioritize later.
