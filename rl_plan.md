# Reinforcement Learning Plan — *Bus*

**Status:** Draft / Iterative
**Purpose:** Define a clear, scoped plan for integrating Reinforcement Learning (RL), specifically self-play with PPO, into the *Bus* game engine.

This document assumes:

* A deterministic, rule-complete, headless game engine already exists
* The full game state is accessible and serializable
* Action legality is enforced by the engine

---

## 1. Objectives

The primary RL objective is to:

> Learn strong, general-purpose policies for playing *Bus* via self-play in a fully observable, deterministic environment.

Secondary objectives:

* Enable rapid experimentation with reward shaping
* Maintain full debuggability of agent decisions
* Support curriculum learning and ablation studies (optional, low priority)

---

## 2. High-Level RL Approach

### 2.1 Learning Paradigm

* **Algorithm:** Proximal Policy Optimization (PPO), possibly others
* **Training mode:** Self-play
* **Policy sharing:** Single shared policy across all players
* **Perspective:** Current-player view (state normalized or indexed by player ID)

This mirrors best practices for symmetric, turn-based board games.

---

## 3. Environment Interface (Gym-Style)

### 3.1 Environment Structure

The RL environment exposes a **single-agent, turn-based interface**:

```text
Agent controls the current player
Environment advances turns internally
```

This avoids multi-agent instability while preserving self-play dynamics.

### 3.2 Step API (Conceptual)

```python
obs, reward, done, info = env.step(action)
```

* `obs`: Observation from the perspective of the current player
* `reward`: Immediate reward for the player who just acted
* `done`: Whether the game has ended
* `info`: Debug metadata (optional)

---

## 4. Observation Design

### 4.1 Guiding Principle

> The observation must be **Markov-complete** and **fully observable**.

No information is hidden from the agent.

### 4.2 Observation Contents

Observations should include:

* **Board Graph State**

  * Node features (zone, buildings, passengers)
  * Edge features (rail ownership counts per player and locations)
* **Bus State**

  * Bus locations per player
  * Bus availability flags
* **Action Board State**

  * Marker placements (all players)
  * Slot occupancy
* **Player State**

  * Buses owned
  * Score
  * Remaining action markers
  * Time Stones taken
* **Global State**

  * Current phase
  * Current player index
  * Time Clock position
  * Time Stones remaining
  * Passengers remaining

Structured tensor or dictionary-based observations are recommended. Should be formatted for easy use by the agent - PyTorch implementation.

---

## 5. Action Space Design

### 5.1 Phase-Dependent Action Spaces

The action space is **conditional on the game phase**:

* Choosing Actions phase
* Resolving Actions phase
* Cleanup phase (no actions)

Each phase exposes a different, mutually exclusive action set.
The resolving actions phase is dependent on where the markers are placed.

---

### 5.2 Choosing Actions Phase

**Action type:** Action marker placement

Actions include:

* `(action_area, slot_index)`
* `PASS`

Constraints are enforced via **action masking**:

* Slot must be empty
* Player must still have markers
* Player must not have passed
* The slot index is deterministic based on the action marker placement, players have no agency to choose the slot index. This may simplify the action space here, as the marker placement from all players should be visible in the game state.

---

### 5.3 Resolving Actions Phase

**Action type:** Contextual resolution decisions

Examples:

* Line Expansion: choose edge / endpoint
* Buildings: choose node / building type
* Vrrooomm!: choose `(passenger, destination)` 
* Passengers: choose the train stations where the passengers are placed
* Time clock: choose to take a time stone and freeze time, or allow time to pass naturally

The engine dynamically enumerates **valid resolution options**, and the agent selects an index into this list.

This guarantees legality and keeps the action space compact.

* Bus and Starting Player actions are resolved automatically - the decision space for these is in the marker placement (choosing actions) phase.

---

## 6. Action Masking (Critical)

At every decision point:

* The environment generates a mask over valid actions
* Invalid actions are never sampled

This is mandatory for PPO or other agent stability and learning efficiency.

---

## 7. Reward Design

### 7.1 Core Rewards (Sparse)

Baseline rewards:

* `+1` per passenger delivered
* `-1` per Time Stone owned (terminal)
* Terminal reward: score differential vs opponents

This establishes correctness but may learn slowly.

---

### 7.2 Reward Shaping (Optional but Recommended after Core Implementation)

Small, aligned shaping rewards improve learning speed:

#### Transport Efficiency

* Small positive reward for increasing reachable valid destinations, perhaps variety of buildings covered, train stations reached, or increasing number of passengers on the agent's line. 
* Small penalty for unused buses during Vrrooomm!

#### Network Quality

* Reward for expanding connected components
* Penalty for creating loops
* Could use some type of graph theoretical metrics to reward the agent
* Diversity of connections

#### Strategic Interaction

* Reward for blocking opponent transport paths
* Reward for taking passengers from opponent's lines (limiting accessible scoring opportunities)
* Penalty for enabling opponent scoring
* Reward for forcing opponent to take time stones (opportunity cost)

Shaping rewards should be **small relative to terminal reward**.

---

## 8. Self-Play Strategy

### 8.1 Shared Policy

* All players act using the same policy
* The policy conditions on the current player index

This enforces symmetry and generality.

---

### 8.2 Opponent Sampling

To prevent training instability:

* Train against a mixture of:

  * Current policy
  * Frozen past checkpoints

This stabilizes learning and avoids policy collapse.

---

## 9. Curriculum Learning

Recommended progression:

1. Reduced map or player count
2. Disable Time Stones
3. Simplified building rules
4. Full ruleset

Curriculum progression should be automated where possible.

---

## 10. Baselines and Validation

Before RL training:

* Implement a random agent
* Implement a simple heuristic agent

Use these to:

* Validate environment correctness
* Establish performance baselines

---

## 11. Debugging and Visualization

Visualization tools should be used to:

* Inspect failed rollouts
* Analyze agent decisions
* Understand reward attribution

Visualization is not part of the observation.

---

## 12. Natural Implementation Progression

1. Wrap engine in Gym-style environment
2. Define observation encoding
3. Implement action masking
4. Add baseline agents
5. Train PPO without shaping
6. Introduce reward shaping
7. Add self-play opponent pool
8. Introduce curriculum learning
9. Iterate and refine

---

## 13. Success Criteria

Early success:

* Agent outperforms random
* Reduced wasted actions
* Consistent passenger delivery

Long-term success:

* Strong self-play equilibria
* Robust play against heuristics
* Interpretable strategic behavior

Eventually:

* RL integration with the current GUI for play against human players (select specific RL policy to play against)
* AI selection and automatic action resolution

---

## 14. Notes

This plan prioritizes:

* Stability over cleverness
* Interpretability over raw performance
* Incremental validation at every step

RL integration should never require rewriting core game logic.
