# GameState.clone() Optimization Plan

## Problem

`GameState.clone()` currently calls `copy.deepcopy(self)`:

```python
# core/game_state.py:293
def clone(self) -> GameState:
    return copy.deepcopy(self)
```

This is called by `GameEngine.clone()` → `BusEnv.clone()` → `AlphaZeroMCTS._build_root()` and `_simulate()` — once per MCTS simulation, so `n_simulations + 1` times per move in self-play. Python's `deepcopy` is extremely slow on complex nested objects: it maintains a memo dict, traverses the entire object graph recursively, and runs `isinstance` dispatch logic on every field. This makes MCTS effectively freeze on the smoke test.

The stack that shows up on Ctrl+C:

```
mcts._build_root / _simulate
  → BusEnv.clone
    → GameEngine.clone
      → GameState.clone
        → copy.deepcopy(self)   ← all CPU time spent here
```

## Key Observation: The Work Is Already Done

`board.py` already has fast hand-written `clone()` methods on every board class:

| Class | Status |
|---|---|
| `BuildingSlot` | ✅ hand-written clone |
| `NodeState` | ✅ hand-written clone |
| `EdgeState` | ✅ hand-written clone |
| `BoardGraph` | ✅ hand-written clone |
| `GlobalState` | ❌ no clone (falls back to deepcopy) |
| `Player` | ❌ no clone |
| `ActionSlot` | ❌ no clone |
| `ActionArea` | ❌ no clone |
| `ActionBoard` | ❌ no clone |
| `Passenger` | ❌ no clone |
| `PassengerManager` | ❌ no clone |
| **`GameState`** | ❌ calls `deepcopy`, ignoring all existing clones |

`GameState.clone()` bypasses all of the existing fast clones and starts deepcopy from scratch. The fix is to wire up `GameState.clone()` to call the existing `board.clone()` chain and add fast clones to the remaining five simple component classes.

## Object Graph

```
GameState
├── board: BoardGraph                     ← already has clone()
│   ├── nodes: dict[NodeId, NodeState]    ← already has clone()
│   │   └── building_slots: list[BuildingSlot]  ← already has clone()
│   ├── edges: dict[EdgeId, EdgeState]    ← already has clone()
│   │   └── rail_segments: list[RailSegment]    (just int, trivial)
│   └── adjacency: dict[NodeId, set[NodeId]]    ← SHARE (immutable topology)
├── players: list[Player]                 ← needs clone()
│   └── network_endpoints: set[int]       (authoritative state, must copy)
├── action_board: ActionBoard             ← needs clone()
│   └── areas: dict[ActionAreaType, ActionArea]  ← needs clone()
│       └── slots: dict[str, ActionSlot]  ← needs clone()
├── passenger_manager: PassengerManager   ← needs clone()
│   └── passengers: dict[int, Passenger]  (trivial: two ints each)
├── global_state: GlobalState             ← needs clone()
│   └── (all primitives + one enum)
└── phase: Phase                          (enum, immutable — share)
```

## Implementation Plan

### 1. `core/board.py` — optimize `BoardGraph.clone()`

The current `adjacency` copy creates a new `set` for every node in the graph on every MCTS clone. Since adjacency is confirmed immutable after board load, share the reference:

**Change** in `BoardGraph.clone()`:
```python
# Before:
new_board.adjacency = {node_id: set(neighbors) for node_id, neighbors in self.adjacency.items()}

# After:
new_board.adjacency = self.adjacency  # immutable topology; safe to share
```

No other changes to `board.py`.

---

### 2. `core/game_state.py` — add `GlobalState.clone()`, fix `GameState.clone()`

#### `GlobalState.clone()`

All fields are primitives or enums (immutable). Add directly after the last method:

```python
def clone(self) -> GlobalState:
    return GlobalState(
        round_number=self.round_number,
        current_player_idx=self.current_player_idx,
        starting_player_idx=self.starting_player_idx,
        time_clock_position=self.time_clock_position,   # enum, immutable
        time_stones_remaining=self.time_stones_remaining,
        current_resolution_area_idx=self.current_resolution_area_idx,
        current_resolution_slot_idx=self.current_resolution_slot_idx,
        game_ended=self.game_ended,
    )
```

#### `GameState.clone()`

Replace the `copy.deepcopy(self)` body:

```python
def clone(self) -> GameState:
    return GameState(
        board=self.board.clone(),
        players=[p.clone() for p in self.players],
        action_board=self.action_board.clone(),
        passenger_manager=self.passenger_manager.clone(),
        global_state=self.global_state.clone(),
        phase=self.phase,           # enum, immutable
    )
```

Remove the `import copy` if it becomes unused (check — `state_hash` uses `json` but not `copy`; `copy` is only used in `clone`, so the import can be dropped).

---

### 3. `core/player.py` — add `Player.clone()`

All fields are primitives except `network_endpoints: set[int]`, which is **authoritative state** actively maintained by `SetupManager` and `LineExpansionResolver`. Must be copied.

```python
def clone(self) -> Player:
    return Player(
        player_id=self.player_id,
        action_markers_remaining=self.action_markers_remaining,
        rail_segments_remaining=self.rail_segments_remaining,
        buses=self.buses,
        score=self.score,
        time_stones=self.time_stones,
        has_passed=self.has_passed,
        markers_placed_this_round=self.markers_placed_this_round,
        network_endpoints=set(self.network_endpoints),
    )
```

---

### 4. `core/action_board.py` — add `ActionSlot.clone()`, `ActionArea.clone()`, `ActionBoard.clone()`

#### `ActionSlot.clone()`

All fields are primitives. Add to `ActionSlot`:

```python
def clone(self) -> ActionSlot:
    return ActionSlot(
        label=self.label,
        player_id=self.player_id,
        placement_order=self.placement_order,
    )
```

#### `ActionArea.clone()`

`area_type` is an enum (immutable), `max_slots` is an int. The `slots` dict needs element-wise cloning. `__post_init__` only reinitializes slots when `not self.slots`, so passing the cloned dict is safe:

```python
def clone(self) -> ActionArea:
    return ActionArea(
        area_type=self.area_type,
        slots={label: slot.clone() for label, slot in self.slots.items()},
        max_slots=self.max_slots,
    )
```

#### `ActionBoard.clone()`

`placement_counter` is an int. `areas` dict needs element-wise cloning. `__post_init__` only reinitializes when `not self.areas`, so passing the cloned dict is safe:

```python
def clone(self) -> ActionBoard:
    return ActionBoard(
        areas={area_type: area.clone() for area_type, area in self.areas.items()},
        placement_counter=self.placement_counter,
    )
```

---

### 5. `core/components.py` — add `PassengerManager.clone()`

`Passenger` has two int fields, so inline the construction rather than adding a separate `Passenger.clone()`. `PassengerManager` has a custom `__init__` (not the dataclass-generated one), so use `object.__new__` to bypass it:

```python
def clone(self) -> PassengerManager:
    new_mgr = object.__new__(PassengerManager)
    new_mgr._next_id = self._next_id
    new_mgr.passengers = {
        pid: Passenger(passenger_id=p.passenger_id, location=p.location)
        for pid, p in self.passengers.items()
    }
    return new_mgr
```

---

## What We Are NOT Changing

- `BusEnv.clone()` — already correctly structured; will automatically benefit since `engine.clone()` → `state.clone()` is now fast.
- `GameEngine.clone()` — already shares `_board` (topology); no change needed.
- `_reward_calculator._stations_connected` — already manually copied in `BusEnv.clone()`; not part of `GameState`.
- `_vrroomm_stage_state` — already manually copied in `BusEnv.clone()`; not part of `GameState`.

## Files Changed (summary)

| File | Changes |
|---|---|
| `core/board.py` | Share `adjacency` reference in `BoardGraph.clone()` |
| `core/game_state.py` | Add `GlobalState.clone()`; replace `GameState.clone()` body; remove `import copy` |
| `core/player.py` | Add `Player.clone()` |
| `core/action_board.py` | Add `ActionSlot.clone()`, `ActionArea.clone()`, `ActionBoard.clone()` |
| `core/components.py` | Add `PassengerManager.clone()` |

## Testing

1. **Existing test suite** — run `python -m pytest tests/` before and after. All tests should pass unchanged, since the clone semantics are identical (deep independent copy).

2. **Smoke test** — `python scripts/train_mcts.py --iterations 2 --games_per_iter 2 --n_simulations 20 --n_workers 1` should complete in well under 5 minutes.

3. **Clone isolation check** — confirm that mutating a clone's fields does not affect the original. The critical invariant is that after `clone = state.clone()`, any mutation to `clone.board`, `clone.players`, `clone.action_board`, `clone.passenger_manager`, or `clone.global_state` is invisible to `state`. This is guaranteed by construction: every mutable container is explicitly copied.

4. **Adjacency sharing sanity check** — confirm `clone.board.adjacency is state.board.adjacency` is `True` and that no gameplay code ever assigns to adjacency (grep for `adjacency =` or `adjacency[` writes).

## Expected Speedup

Python's `deepcopy` pays a fixed overhead per object visited (memo dict lookup, `isinstance` dispatch, `__reduce__` checks). On a 4-player game board with ~100 nodes, ~200 edges, 7 action areas, and ~20 passengers, the object graph has on the order of 1,000–2,000 Python objects to traverse.

The hand-written clone visits only the mutable containers (no dispatch overhead, no memo dict) and allocates objects directly via constructors. In practice, hand-written clones of equivalent Python data structures are typically **20–50× faster** than `deepcopy`.

With `n_simulations=20`, each move runs 21 clone operations. The cumulative effect across a full game (hundreds of moves) is the difference between a smoke test completing in seconds vs. appearing frozen.

## Implementation Checklist

- [x] `core/board.py` — Share `adjacency` reference in `BoardGraph.clone()` (was copying a new `set` per node)
- [x] `core/board.py` — Remove dead `import copy` (was unused; none of the board clone methods call `copy.*`)
- [x] `core/game_state.py` — Add `GlobalState.clone()` (all primitives + one enum; direct constructor call)
- [x] `core/game_state.py` — Replace `GameState.clone()` body (was `copy.deepcopy(self)`; now delegates to component clones)
- [x] `core/game_state.py` — Remove `import copy` (now unused after clone replacement)
- [x] `core/player.py` — Add `Player.clone()` (`network_endpoints` copied with `set()`; all other fields are primitives)
- [x] `core/action_board.py` — Add `ActionSlot.clone()` (all primitives)
- [x] `core/action_board.py` — Add `ActionArea.clone()` (slots dict cloned element-wise; `__post_init__` guard respected)
- [x] `core/action_board.py` — Add `ActionBoard.clone()` (areas dict cloned element-wise; `__post_init__` guard respected)
- [x] `core/components.py` — Add `PassengerManager.clone()` (`object.__new__` bypasses custom `__init__`; passengers reconstructed inline)
- [x] Test suite passes (`python -m pytest tests/`) — 583 passed, 16 pre-existing failures (reward/obs config values; unrelated to clone)
- [ ] Smoke test completes (`python scripts/train_mcts.py --iterations 2 --games_per_iter 2 --n_simulations 20 --n_workers 1`)

## Resolver Clone Fix

Identified gap: `BusEnv.clone()` did not copy `_resolver`, so clones made mid-resolution
(e.g. between rail segment placements within a single line-expansion slot) would reconstruct
a fresh `ActionResolver` via `start_resolution()`, losing within-slot progress and potentially
re-visiting already-resolved areas.

- [x] `engine/action_resolver.py` — Add `ActionResolver.clone(new_state)`:
  - Creates fresh `ActionResolver(new_state)` then overrides `_context` with a cloned `ResolutionContext`
  - `current_slot` remapped to the equivalent slot in the cloned action board (same area + label)
  - Only the **active** sub-resolver is cloned; completed and pending ones stay `None`
  - `LineExpansionResolver`: copy `_current_slot_idx`, `_segments_placed_in_current_slot`
  - `PassengersResolver`: copy `_current_slot_idx`
  - `BuildingsResolver`: copy `_current_slot_idx`, `_buildings_placed_in_current_slot`
  - `VrrooommResolver`: create normally so `__init__` reconstructs `_occupied_slots` from cloned board; then copy `_current_slot_idx`, `_deliveries_in_current_slot`
  - `_area_results` discarded — not needed for simulation continuity
- [x] `rl/bus_env.py` — `BusEnv.clone()`: call `self._resolver.clone(new_env._engine.state)` when `_resolver is not None`, immediately after `_engine` is cloned so the new resolver references the correct cloned state
- [x] Test suite passes with resolver fix — 583 passed, same 16 pre-existing failures
