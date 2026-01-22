# Bus Board Game Engine - Implementation Plan

## Overview
Build a headless, deterministic Python implementation of the board game *Bus* with:
- Full rule enforcement via a graph-based board representation
- RL-ready state (Gymnasium interface with action masking for PPO)
- Clean separation: game engine / RL wrapper / visualization

## Clarified Requirements Summary

| Aspect | Decision |
|--------|----------|
| Player count | 3-5 (variable) |
| Board data | User will provide; schema defined together |
| Rail sharing | Allowed only if no empty edges at endpoint OR endpoints touch another player's |
| Bus movement | One passenger per bus per Vrroomm!, unlimited travel distance; passengers stay at destination after delivery |
| Building placement | Mandatory inner-first (zones A->B->C->D); quantity = M#oB - slot_index |
| Passing | Once passed, out for the round |
| Setup | First player places 2 buildings (inner zone), then clockwise. Then each player places 1 rail segment (any edge), then reverse order places 1 more (normal expansion rules). 4 passengers start at "central park" nodes. |
| Scoring | Buildings are unowned; deliver to any matching type |
| Passengers action | Quantity = M#oB - slot_index; player chooses station distribution |
| Time clock | Starts on House; order: House -> Office -> Pub -> House (clockwise) |

---

## Project Structure

```
bus/
├── __init__.py
├── core/                      # Data models (immutable topology, mutable state)
│   ├── constants.py           # Enums: Zone, BuildingType, Phase, ActionAreaType
│   ├── board.py               # BoardGraph, NodeState, EdgeState, NodeId, EdgeId
│   ├── player.py              # Player model with resources
│   ├── action_board.py        # ActionArea, ActionSlot, marker tracking
│   ├── components.py          # Passenger, Building, RailSegment
│   └── game_state.py          # GameState - single source of truth
├── engine/                    # Game logic
│   ├── game_engine.py         # Main orchestrator: step(), reset(), get_valid_actions()
│   ├── phase_machine.py       # Phase transitions state machine
│   ├── action_resolver.py     # Dispatcher to individual resolvers
│   ├── setup.py               # Initial game setup logic
│   ├── validators.py          # State invariant checks
│   └── resolvers/             # One file per action type
│       ├── line_expansion.py
│       ├── buses.py
│       ├── passengers.py
│       ├── buildings.py
│       ├── time_clock.py
│       ├── vrroomm.py
│       └── starting_player.py
├── rl/                        # Reinforcement learning interface
│   ├── bus_env.py             # Gymnasium environment
│   ├── observation.py         # State -> tensor encoding
│   ├── action_space.py        # Action encoding/decoding
│   ├── action_masking.py      # Valid action mask generation
│   └── reward.py              # Reward functions
├── data/                      # Board definitions
│   ├── board_schema.json      # JSON schema for board files
│   ├── default_board.json     # Standard Bus board
│   └── loader.py              # Board loading/validation
├── visualization/             # Debug/inspection tools
│   ├── renderer.py            # NetworkX + matplotlib
│   └── formatters.py          # Text-based state display
└── tests/                     # Test suite
```

---

## Key Components

### 1. Core Data Models

**GameState** (single source of truth):
- `board: BoardGraph` - nodes, edges, adjacency (static topology, mutable occupancy)
- `players: list[Player]` - resources, scores, markers
- `action_board: ActionBoard` - marker placements per area
- `global_state: GlobalState` - time clock, round, current player, resolution tracking
- `passengers: dict[int, Passenger]` - all passengers with unique IDs
- `phase: Phase` - current game phase

**BoardGraph**:
- Nodes have: building slots (each with its own zone), passengers present, is_train_station, is_central_park
- Building slots have: zone (A/B/C/D), optional building placed. A node can have slots from multiple zones.
- Edges have: endpoints, list of rail segments (player ownership)
- Adjacency dict for graph traversal
- Methods: `get_player_network_endpoints()`, `get_reachable_nodes()`

### 2. Phase State Machine

**Setup (once at game start):**
`SETUP_BUILDINGS` -> `SETUP_RAILS_FORWARD` -> `SETUP_RAILS_REVERSE` -> `CHOOSING_ACTIONS` (where game begins)

**Main game loop:**
`CHOOSING_ACTIONS` -> `RESOLVING_ACTIONS` -> `CLEANUP` -> (back to `CHOOSING_ACTIONS` or `GAME_OVER`)

Setup phases execute exactly once at the beginning. The loop starts at `CHOOSING_ACTIONS` for all subsequent rounds.

### 3. Action Resolution Order

Fixed: Line Expansion -> Buses -> Passengers -> Buildings -> Time Clock -> Vrroomm! -> Starting Player

Within each area: resolve slots left-to-right.

### 4. RL Interface

- **Observation**: Dict space with node features, edge features, player features, action board state, global state
- **Action space**: Discrete (flat index covering PASS, PLACE markers, and resolution choices)
- **Action masking**: Boolean mask generated from `get_valid_actions()`; required for MaskablePPO

---

## Board Data Schema (JSON)

```json
{
  "nodes": [
    {
      "id": 0,
      "building_slots": ["A", "B"],
      "is_train_station": false,
      "is_central_park": false,
      "position": {"x": 5, "y": 3}
    }
  ],
  "edges": [
    [0, 1]
  ]
}
```

**Node fields:**
- `id`: Unique integer identifier
- `building_slots`: Array of zones (e.g., `["A", "B"]` means one Zone A slot and one Zone B slot)
- `is_train_station`: true for the 2 passenger spawn locations (empty building_slots)
- `is_central_park`: true for the 4 starting passenger locations
- `position`: Object with `x`, `y` coordinates for visualization

---

## Implementation Phases

### Phase 1: Core Models
1. `core/constants.py` - all enums and constants
2. `core/board.py` - NodeId, EdgeId, NodeState, EdgeState, BoardGraph
3. `core/player.py` - Player with resource tracking
4. `core/components.py` - Passenger, Building, RailSegment
5. `core/action_board.py` - ActionSlot, ActionArea, ActionBoard
6. `core/game_state.py` - GameState with clone/serialize/hash

### Phase 2: Board Data
1. `data/board_schema.json` - JSON schema
2. `data/loader.py` - load and validate board JSON
3. Work with user to define `default_board.json`

### Phase 3: Engine Foundation
1. `engine/phase_machine.py` - phase transitions
2. `engine/setup.py` - initial game setup
3. `engine/game_engine.py` - step(), reset(), get_valid_actions()

### Phase 4: Action Resolvers (in order of complexity)
1. `resolvers/starting_player.py`
2. `resolvers/buses.py`
3. `resolvers/time_clock.py`
4. `resolvers/passengers.py`
5. `resolvers/buildings.py`
6. `resolvers/line_expansion.py`
7. `resolvers/vrroomm.py`
8. `engine/action_resolver.py` - dispatcher

### Phase 5: RL Integration
1. `rl/observation.py` - encode state to tensors
2. `rl/action_space.py` - action encoding/decoding
3. `rl/action_masking.py` - generate valid action masks
4. `rl/reward.py` - reward calculation
5. `rl/bus_env.py` - Gymnasium environment

### Phase 6: Visualization
1. `visualization/renderer.py` - NetworkX + matplotlib
2. `visualization/formatters.py` - text output

---

## Verification Plan

1. **Unit tests** for each core model and resolver
2. **Integration tests**: run full random games to completion
3. **Property tests**: state invariants hold after any valid action sequence
4. **RL smoke test**: run MaskablePPO training for a few episodes to verify interface works

---

## Critical Files to Implement First

1. `bus/core/constants.py` - foundation for all other modules
2. `bus/core/board.py` - graph model
3. `bus/core/game_state.py` - central state container
4. `bus/engine/game_engine.py` - main interface
5. `bus/data/default_board.json` - need to define with user

---

## Next Steps (TODOs)

1. [x] Create project directory structure
2. [x] Implement `core/constants.py` with enums and constants
3. [x] Implement `core/board.py` with graph model
4. [x] Implement `core/player.py` with resource tracking
5. [x] Implement `core/components.py` (Passenger, PassengerManager)
6. [x] Implement `core/action_board.py`
7. [x] Implement `core/game_state.py`
8. [x] Define board topology with user and create `default_board.json`
9. [x] Implement `data/board_schema.json` - JSON schema for validation
10. [x] Implement `data/loader.py` - load and validate board JSON into BoardGraph

---

## Immediate Next Step

Phase 1 (Core Models), Phase 2 (Board Data), and Phase 3 (Engine Foundation) are complete.

**Phase 4: Action Resolvers** (in order of complexity)
- [x] Implement `resolvers/starting_player.py` - assign next round's starting player
- [x] Implement `resolvers/buses.py` - gain additional buses
- [x] Implement `resolvers/time_clock.py` - advance clock or take time stone (player choice)
- [x] Implement `resolvers/passengers.py` - spawn passengers at train stations (player choice for distribution)
- [x] Implement `resolvers/buildings.py` - place buildings (inner-first rule, player choice for type/location)
- [x] Implement `resolvers/line_expansion.py` - extend rail networks (player choice for edges)
- [x] Implement `resolvers/vrroomm.py` - move passengers and score points (player choice for routing)
- [x] Implement `engine/action_resolver.py` - dispatcher to route resolution actions

### Resolver Implementation Notes

**Completed Resolvers (Phase 4a):**

1. **StartingPlayerResolver** - Simple, no player choice. Whoever placed the marker becomes starting player.

2. **BusesResolver** - Simple, no player choice. Player gains 1 bus (up to MAX_BUSES=5). Immediately updates M#oB.

3. **TimeClockResolver** - Player choice between:
   - ADVANCE_CLOCK: Move clock to next building type (House → Office → Pub → House)
   - STOP_CLOCK: Take a time stone (-1 point at game end). If last stone, game ends.

   - Passengers to spawn = M#oB - slot_index (minimum 1)
   - Player chooses how to distribute passengers between train stations
   - Supports iterative slot-by-slot resolution or batch resolve_all()

**Completed Resolvers (Phase 4b):**

5. **BuildingsResolver** - Player choice for type and location:
   - Buildings to place = M#oB - slot_index (minimum 1)
   - Must place in innermost available zone first (A → B → C → D)
   - Player chooses building type (House, Office, Pub) for each placement
   - Supports iterative building-by-building placement or batch resolve_all()
   - Triggers game end condition if all building slots are filled

6. **LineExpansionResolver** - Player choice for edges:
   - Segments to place = M#oB - slot_index (minimum 1)
   - Must extend from network endpoints
   - Rail sharing rules:
     - Can only share if no empty edges at any endpoint, OR
     - One of player's endpoints touches another player's endpoint
   - Prefers empty edges over shared edges in default resolution
   - Tracks player rail inventory and decrements on placement

**Completed Resolvers (Phase 4c):**

7. **VrrooommResolver** - Player choice for passenger routing:
   - One passenger per bus can be transported per Vrroomm! action
   - Passengers can travel unlimited distance on player's rail network
   - Only deliver to buildings matching current time clock type
   - Score 1 point per delivery
   - Delivered passengers occupy building slot temporarily (prevents double-delivery)
   - Passengers already at nodes with matching buildings auto-occupy those slots
   - Building slot occupancy cleared after all Vrroomm! actions complete

8. **ActionResolver** - Dispatcher for all resolution actions:
   - Coordinates resolution of all action areas in correct order
   - Provides step-by-step resolution with player input or batch resolve_all()
   - Tracks resolution context (current area, slot, status)
   - Detects game end conditions (time stones, building slots filled)
   - Creates appropriate resolver instances on demand

---

## Phase 4 Complete

All action resolvers have been implemented:
- StartingPlayerResolver
- BusesResolver
- TimeClockResolver
- PassengersResolver
- BuildingsResolver
- LineExpansionResolver
- VrrooommResolver
- ActionResolver (dispatcher)

**Playable Driver (Interactive CLI) - COMPLETE**
- [x] Implemented `engine/driver.py` - main loop and user input handling:
  - Game loop `while not game_over`
  - State display with text formatters (TextRenderer class)
  - Action prompts for the Setup phase (initial placements of buildings and rail segments)
  - Action prompts for Choosing phase (marker placement and passing)
  - Action prompts for Resolving phase (all 7 action areas)
  - Error handling and invalid action feedback
  - **GUI-ready architecture** with abstract interfaces:
    - `GameRenderer` (ABC) - swap TextRenderer for a GUI renderer
    - `ActionPrompter` (ABC) - swap TextPrompter for GUI event handling
    - `GameDriver` - orchestrates game loop, delegates to renderer/prompter

### Driver Architecture (GUI-Ready)

The driver uses a clean separation of concerns to enable easy GUI replacement:

```
GameDriver
├── GameEngine (game logic)
├── GameRenderer (abstract)
│   └── TextRenderer (CLI implementation)
└── ActionPrompter (abstract)
    └── TextPrompter (CLI implementation)
```

To create a GUI version:
1. Implement `GameRenderer` with your GUI framework
2. Implement `ActionPrompter` with GUI event handling
3. Pass them to `GameDriver(renderer=MyGUIRenderer(), prompter=MyGUIPrompter())`

**Usage:**
```bash
# Run the CLI game
python -m engine.driver

# Or from code:
from engine.driver import GameDriver
driver = GameDriver(num_players=4)
driver.run()
```

---

**Phase 5: RL Integration**
- [ ] Implement `rl/observation.py` - encode state to tensors
- [ ] Implement `rl/action_space.py` - action encoding/decoding
- [ ] Implement `rl/action_masking.py` - generate valid action masks
- [ ] Implement `rl/reward.py` - reward calculation
- [ ] Implement `rl/bus_env.py` - Gymnasium environment

---

**Phase 6: Visualization & GUI**

### 6.1 Current Visualization (Complete)
- [x] `data/graph_vis.py` - NetworkX + matplotlib board visualization
  - Renders board topology, buildings, passengers, rail networks
  - Supports player-specific network highlighting
  - Color-coded zones, building types, and player ownership

### 6.2 Future: Full GUI Application

A fully-fledged GUI is planned for human gameplay. Key requirements:

**Framework Options (TBD):**
- PyQt/PySide6 - Native cross-platform desktop app - Let's start with this
- Could move to web-based (Flask/FastAPI + React/Vue) - Browser-based interface

**GUI Features (Planned):**
- Interactive board visualization (click to select nodes/edges)
- Either drag-and-drop for building/rail placement or click based selection
- Passenger movement during Vrroomm! phase, updated location (not necessarily animated)
- Passenger on building slot visualization during Vrroomm! phase
- Real-time game state updates
- Player turn indicators and action highlighting
- Action history log
- Game over screen and scoring display
- Visualization of graph-based board (left side of the screen) concurrent with the action board (right side of the screen)
- Graph-based board should show rail lines, buildings (with typing), and passenger locations
- Marker visualization on the action board
- Number of buses per player displayed on/near the Buses section of the action board

**Additional Features (Optional):**
- Undo/redo support (using engine's clone() capability)
- Save/load game functionality (using state serialization)
- Sound effects and visual polish

**Integration Points:**
- `GameRenderer` implementation for rendering game state to GUI widgets
- `ActionPrompter` implementation for converting GUI events to action choices
- Use existing `data/graph_vis.py` as reference for board rendering logic
- Leverage `GameState.to_dict()` for save/load
- Use `GameEngine.clone()` for undo functionality

**Implementation Order (Suggested):**
1. Basic board rendering with interactive nodes
2. Setup phase UI (building/rail placement)
3. Choosing actions phase UI (action board interaction)
4. Resolution phase UI (player choices with visual feedback)
5. Game over screen and scoring display
6. Polish: animations, sound, save/load
