# PROJECT PLAN — Bus (Digital Board Game + RL Environment)

**Version:** 1.1  
**Status:** Living document  
**Primary Goal:** Deterministic, rule-complete game engine with a fully observable, RL-ready state representation (gym-like wrapper)

---

## 1. Project Intent

The objective of this project is to build a **headless, deterministic digital implementation** of the board game *Bus* that:

- Fully enforces the official game rules
- Uses a graph-based representation of the board
- Exposes a complete, structured game state suitable for Reinforcement Learning (RL)
- Cleanly separates game logic from UI and learning agents

The engine must support:
- Human play (via a future UI layer)
- PPO-based RL agents
- Self-play training
- Full state inspection, cloning, and serialization
- Gym-style interface for RL agents

**Non-goals (initially):**
- Online multiplayer
- Pixel-based RL training
- Visual polish or animations

---

## 2. Technology Constraints

- **Core engine:** Python
- **RL stack:** Python (Gym-style interface, PyTorch-compatible)
- **UI:** Deferred; Python or Web-based acceptable
- **Map:** Fixed, static topology
- **Observability:** Fully observable
- **Stochasticity:** None inherent; only arises from player decisions

---

## 3. Design Principles

1. **Single source of truth**  
   All state changes occur through the game engine.

2. **Explicit state**  
   No hidden or derived gameplay information.

3. **Action legality is enforced, not learned**  
   Illegal actions are masked and never executed.

4. **Immutable structure, mutable occupancy**  
   Board topology never changes; pieces do.

5. **Engine-first development**  
   UI and RL layers wrap the engine rather than reimplement logic.

---

## 4. Board Representation

### 4.1 Graph Model

The board is represented as a **static attributed graph**:

- **Nodes** represent intersections
- **Edges** represent streets
- **Rail segments** represent player-owned elements on edges

#### Node State
Each node stores:
- Building slots (each slot has its own zone: A, B, C, or D)
- Buildings placed in each slot (type, occupied - only during Vrrooomm! action)
- Passengers currently at the intersection

Note: Nodes themselves do not have a zone. A single node may contain building slots from different zones (e.g., one slot in Zone A and one slot in Zone B).

#### Edge State
Each edge stores:
- Endpoints (node IDs)
- A list of rail segments (one per player if applicable)

Adjacency is defined once and never modified.

---

## 5. Core Game State (RL-Complete)

The `GameState` must be **Markov-complete**: it fully defines the game at any moment.

### 5.1 Conceptual Structure

```
GameState
├── BoardGraph
│   ├── Nodes
│   ├── Edges
│   └── Adjacency (static)
├── Players
│   ├── buses
│   ├── score
│   ├── remaining action markers
│   └── rail inventory
├── Action Board
│   ├── action areas
│   ├── marker placements
│   └── placement order
├── Global
│   ├── current player
│   ├── start player
│   ├── phase
│   ├── time clock position
│   ├── time stones remaining
│   └── round number
```

The state must be:
- Serializable
- Clonable
- Hashable (for debugging and RL rollouts)

---

## 6. Game Phases (Explicit State Machine)

The game progresses through explicit phases:

1. **Choosing Actions**
2. **Resolving Actions**
3. **Cleanup**
4. **Game Over**

The current phase is always part of the game state.

---

## 6.1 Core Game Rules and Turn Structure (Engine-Oriented)

This section defines the authoritative rules of Bus in a form suitable for implementation in a deterministic game engine and for use by AI agents. All rules described here are fully observable and must be representable within the game state.

### 6.1.1 Game Overview

Bus is a deterministic, turn-based, multi-player worker-placement and network-building game played over a fixed map represented as a graph. Players compete to score points by efficiently transporting passengers to buildings of the currently active type using their owned rail networks and buses.

The game proceeds over a sequence of rounds until one of the defined end-game conditions is met.

There is no randomness after setup. All variability arises solely from player actions.

Non player components:
5 time stones, 90 buildings tiles (30 pubs, 30 houses, 30 offices), starting player token, 15 passengers

### 6.1.2 Players (3-5 players) and Components (Logical Representation)

The game supports 3, 4, or 5 players (variable per game).

Each player has the following resources, all of which must be tracked in the game state:

Action markers (20 total, non-recoverable)

Rail segments (25 per player, player-owned, non-recoverable)

Buses (initially 1, increases over time, maximum of 5 per player)

Score marker

Action markers placed this round

Owned rail network (edges on the graph)

Start Player token (exactly one player at a time)

### 6.1.3 Board and Graph Model

The board is represented as a fixed graph:

Nodes represent intersections and building locations. Two nodes represent the train stations, where passengers are placed during the Passenger action, and hold no buildings. (Note: "bus stations" in earlier drafts was a typo; these are train stations.) 

Edges represent streets where rail segments may be built.

Multiple rail segments may exist on the same edge, owned by different players, subject to placement restrictions in the Line Expansion action.

***Nodes*** contain:

Building slots (each slot has its own zone: A, B, C, or D; a node may have slots from multiple zones)

Placed buildings present in each slot (House, Office, Pub)

Passengers present

Note: Nodes do not have zones; building slots do. A single node can have slots from different zones.

***Buildings*** may contain:

Passenger, during the Vrrooomm! action resolution

***Passengers***:

Should have a unique ID, for transport reasons (unique, immutable)

Should have a current location (node ID)

***Edges*** contain:

Zero or more rail segments

Ownership metadata per segment (player ID, can be multiple per edge or empty)

This graph is static in topology but dynamic in state.


### 6.1.4 Time Clock and Building Types

The Time Clock points to exactly one building type at all times:

House

Office

Pub

The active building type:

Restricts where passengers may score

Determines which buildings accept passengers during movement

The Time Clock advances naturally in its phase and is otherwise deterministic unless paused by the optional Stop Time action.

### 6.1.5 Maximum Number of Buses (M#oB)

Many actions scale based on the Maximum Number of Buses (M#oB), defined as:

The highest number of buses owned by any player at the current moment.

M#oB is dynamic and may change mid-round.

Any action that depends on M#oB must query its value at resolution time, not placement time.

This value is global and must be recomputed whenever a player gains a bus.

### 6.1.6 Game Phases Per Round

Each round consists of three phases, executed strictly in order:

***Phase 1: Choosing Actions***

Players take turns placing action markers into action areas on the board.

Rules:

Turn order starts with the Start Player and proceeds clockwise.

On a player’s turn, they must place exactly one action marker or pass (if allowed).

Each player must place at least two action markers per round, if possible.

Action markers:

Are placed in action areas with ordered slots (A, B, C, …)

Must occupy the lowest available lettered slot in that area

Are never recovered once used

Some action areas allow only one marker total

Players may place multiple markers in the same action area if space allows

This phase ends when all players have passed.

Once a player passes, they are out for the remainder of the round and cannot place any more markers.

Engine note:
This phase is purely about reservation of future actions. The only state changes that occur here are marker placement, which can be important for policy learning.

***Phase 2: Resolving Actions***

Action areas are resolved in a fixed global order, independent of placement order:

Line Expansion

Buses

Passengers

Buildings

Time Clock

Vrrooomm!

Starting Player

Within each action area:

Action markers are resolved left to right

Letter labels (A, B, C…) only determine placement order, not execution order

Players must perform actions to the fullest extent possible, with the exception of the time clock action, which may be stopped.

Actions may require contextual decisions (e.g., where to build, which passengers to move)

Engine note:
This phase is deterministic given:

Current state

Marker placement

Player decisions during resolution
It is the primary phase requiring branching player/RL decisions.

Phase 3: Clean-Up

All action markers placed this round are permanently removed from the game.

If only one player has any action markers remaining, the game ends.

Otherwise, a new round begins with Choosing Actions.

### 6.1.7 Action Areas (High-Level Semantics)

Each action area corresponds to a specific category of state modification:

***Line Expansion***

Extend a player’s rail network from its endpoints

Number of segments depends on M#oB and slot letter (A=M#oB, B=M#oB-1, etc.)

Special rules apply for shared streets and merging endpoints

***Buses***

Gain additional buses

May immediately increase M#oB

***Passengers***

Introduce new passengers at train stations

Quantity scales with M#oB (decreases for each subsequent resolution)

***Buildings***

Place buildings into node building slots

Slot availability follows a strict global ordering (1 → 2 → 3 → 4, innermost to outermost)

***Time Clock***

Either advance time clockwise (default) or stop time and take a Time Stone

Time Clock order is fixed: House -> Office -> Pub -> House (clockwise). Clock starts on House.

Time Stones reduce final score by 1 point each.

***Vrrooomm!***

Move passengers along owned rail networks

Score points by delivering passengers to buildings matching the Time Clock - 1 point per passenger delivered.

Passengers already present at a node where the matching Time Clock building is present do not move, and are considered to occupy the building slot during this phase. 

Passengers may be moved to any node where a Time Clock matching building is present, and the building is not occupied by a passenger during this phase.

***Starting Player***

Determine next round’s Start Player

Each of these areas is expanded in detail in the following section.

### 6.1.8 End Game Conditions

The game ends when any one of the following occurs:

The last Time Stone is taken

Only one player has action markers remaining

All building locations on the board are filled

Final scoring:

Each Time Stone is worth −1 point

Highest score wins

Ties are broken by earliest arrival at the tied score

### 6.1.9 Initial Game Setup

The game begins with a setup phase (executed once, before the main game loop):

1. **Building placement (clockwise):**
   - The first player places 2 buildings of their choice in inner zone (Zone A) locations
   - Proceeding clockwise, each other player then places 2 buildings in inner zone locations

2. **First rail segment placement (player order):**
   - In player order, each player places 1 rail segment on any edge
   - Exception: During this initial placement, players may place on edges that already have other players' rails

3. **Second rail segment placement (reverse order):**
   - In reverse player order, each player places 1 rail segment
   - This segment must extend from one endpoint of their first segment
   - Normal line expansion rules apply (respect the usual rules for parallel placement)

4. **Initial passengers:**
   - 1 passenger starts at each "central park" location (4 passengers total)
   - Central park locations are specific nodes defined in the board topology

After setup completes, the game enters the main loop starting with the Choosing Actions phase.

### 6.1.10 Implications for Engine and RL Design

The game is fully observable and deterministic

All randomness comes from player policies

The state must encode:

Marker placement

Action phase

Graph occupancy

Passenger locations (possibly encompassed in graph occupancy)

Time Clock state

Action legality depends on:

Current phase

Marker placement

Graph connectivity

M#oB

This structure is intentionally compatible with:

Rule-based agents

PPO-style RL with action masking

Self-play training regimes

---

## 7. Action Board Design

The action board is a control structure separate from the board graph.

### 7.1 Action Areas

| Area | Slots | Notes |
|----|----|----|
| Line Expansion | F–A | Evaluated left → right, A is on the right |
| Buses | A | Single slot |
| Passengers | A–F | |
| Buildings | F–A | |
| Time Clock | A | Optional resolution|
| Vrrooomm! | A–F | |
| Starting Player | A | |

Each slot tracks:
- Player ID
- Placement order

---

## 8. Player Actions (Formalized)

Player decisions occur in **two distinct phases**, which should be reflected in the RL action space design.

---

## 9. Action Space Design (Critical for RL)

### 9.1 High-Level Structure

The RL action space is **phase-dependent**. At any point, the agent is only allowed to choose from actions valid for the current phase.

This strongly suggests **separate action subspaces** per phase, with shared policy weights if desired.

---

### 9.2 Choosing Actions Phase — Action Space

**Decision Type:** Strategic placement of action markers

#### Example of decision point:
Player 3 has 3 action markers remaining. The action board has the following markers:
- Line Expansion: P1(A), P2(B), P3(C)
- Buses: P1(A)
- Passengers: P3(A), P2(B), P1(C)
- Buildings: P1(A), P2(B)
- Time Clock: P1(A)
- Vrrooomm!: P1(A), P2(B), P3(C)
- Starting Player: P1(A)

Player 3 must decide where to place an action marker, respecting the slot placement order (A, then B, then C, ...) and maximal number of markers per action area. May also pass if they have no more markers to place or choose not to place any more markers (2 markers minimum per round).

**Available Actions:**
- Place an action marker in:
  - A specific action area
  - A specific empty slot in that area, respecting the placement order (A, then B, then C, ...)
- Pass

**Constraints (enforced via masking):**
- Slot must be empty
- Slot must be the next available slot in the placement order (A, then B, then C, ...)
- Area must allow multiple markers or be empty (for single-slot areas)
- Player must place at least 2 markers per round (if markers remain)
- Player cannot act after passing
- Player cannot act if out of markers

**Suggested Encoding:**

Each possible `(area, slot)` combination is a discrete action

Example:
```
ACTION = (AREA_ID, SLOT_INDEX)
PASS = separate discrete action
```

Mask invalid `(area, slot)` pairs dynamically.

---

### 9.3 Resolving Actions Phase — Action Space

**Decision Type:** Tactical choices during mandatory action resolution

Not all action resolutions require player choice. The action space here is **contextual** and often smaller.

#### Examples of decision points:

- **Line Expansion:**
  - Choose which valid edge to extend from
  - Choose which endpoint when multiple are available

- **Passengers:**
  - Choose which station(s) receive new passengers

- **Buildings:**
  - Choose building type
  - Choose eligible node among valid slots

- **Vrrooomm!:**
  - Choose which passengers to move with buses
  - Choose destination buildings (matching time clock)

**Suggested Encoding:**

Use **context-specific action spaces**, generated dynamically:

```
ACTION = index into list of valid resolution options
```

The engine supplies:
- A list of valid resolution actions
- A mapping from indices to concrete state transitions

This keeps the policy compact and legality guaranteed.

---

### 9.4 Cleanup Phase

No player decisions. Fully deterministic.

---

## 10. Action Resolution Logic (Design Notes)

### 10.1 Line Expansion
- Number of segments determined by M#oB and slot letter (A=M#oB, B=M#oB-1, etc.)
- Must extend from an endpoint of the player's existing network
- **Rail sharing rules:**
  - A player may only place on an edge that already has another player's rail segment if:
    1. There are no empty edges available at one of their endpoints, OR
    2. One of their endpoints is at the same node as another player's endpoint (endpoints touching)
  - In both cases, subsequent segments in the same action should generally be placed on empty edges when available
- Segments within one action can alternate between endpoints (extend from endpoint A, then endpoint B, etc.)
- Special case: Loop creation (requires logical endpoint tracking)

### 10.2 Buses
- Increment player bus count
- Immediately update M#oB

### 10.3 Passengers
- Spawn at train stations
- Quantity: M#oB - slot_index (Slot A = M#oB, Slot B = M#oB-1, etc.)
- Player chooses how to distribute spawned passengers between the two train stations

### 10.4 Buildings
- Quantity: M#oB - slot_index (Slot A = M#oB buildings, Slot B = M#oB-1, etc.)
- **Mandatory inner-first placement:** Must place on lowest (innermost) available building slot globally by zone (A -> B -> C -> D). Cannot place in a slot of an outer zone until all slots of inner zones are filled.
- Enforce one building per slot
- Buildings are unowned once placed (any player can deliver passengers to any building)
- Trigger end-game condition if all building slots are filled

### 10.5 Time Clock
- Optional
- Either advance clock or take time stone
- Immediate game end if last stone is taken

### 10.6 Vrrooomm!
- For each marker:
  - Move passengers along player's rail network
  - One passenger per bus (each bus can transport at most one passenger per Vrroomm! action)
  - Passengers can travel unlimited distance along the player's connected rail network
  - Only to buildings matching time clock type (deliver to any matching building, regardless of who placed it)
  - Score immediately (1 point per delivery)
  - Delivered passengers remain at the destination node (not removed from game)
  - During Vrroomm! resolution, delivered passengers temporarily occupy the building slot, preventing additional deliveries to that slot until all Vrroomm! actions complete

### 10.7 Starting Player
- Assign start player for next round

---

## 11. RL Environment Design

### 11.1 Observation

Structured, non-visual observation including:
- Node features (building slot zones and occupancy, passengers present, is_station, is_park)
- Edge features (rail ownership per player)
- Player features (buses, markers, score)
- Action board state
- Global state (phase, time clock, stones)

### 11.2 Action Masking

At every decision point:
- Generate valid actions
- Mask all invalid actions

This is mandatory for PPO stability and efficient learning.

### 11.3 Reward Design (Initial)

- +1 per passenger delivered
- −1 per time stone owned
- Terminal reward:
  - Score differential
  - Optional win/loss bonus

---

## 12. Visualization & Debugging Support

Visualization is an explicit non-UI requirement to support:
- Rule validation
- Debugging complex action resolution (especially Line Expansion and Vrrooomm!)
- Human inspection of RL rollouts

This visualization layer is **not authoritative** and must never mutate game state.

### 12.1 Visualization Goals

The visualization system should allow developers to:
- See all nodes and their building slots (with zone labels per slot)
- Inspect buildings at each slot (type, occupancy during Vrroomm!)
- Inspect passengers at nodes
- See edges between nodes
- See how many rail segments exist on an edge
- See which players own rail segments on each edge
- Optionally highlight a single player’s rail network

### 12.2 Recommended Tooling

**Primary recommendation:** `networkx` + `matplotlib`

Rationale:
- Native graph abstraction (nodes + edges)
- Easy attachment of node/edge attributes
- Flexible enough for debugging visuals
- Zero dependency on UI framework

Alternative tools (optional, later):
- PyGraphviz (static diagrams)
- D3.js (web-based visualization, post-engine)

### 12.3 Graph Visualization Model

The visualization graph mirrors the BoardGraph but is read-only.

#### Node rendering suggestions:
- Node label: node ID
- Node annotation:
  - building slots with zone labels (e.g., "A: Pub, B: empty")
  - passenger count
- Position: use x, y coordinates from board data

#### Edge rendering suggestions:
- Single edge drawn between nodes
- Edge label includes:
  - number of rail segments
  - per-player ownership (color-coded or text)

Example edge label:
```
P0, P1 (2 rails)
```

### 12.4 Player-Specific Network Views

The visualization layer should support filtering:
- Show only edges owned by a specific player
- Highlight endpoints (logical rail ends)
- Highlight reachable nodes for Vrrooomm! actions

This is especially valuable for:
- Verifying line-extension legality
- Debugging passenger movement

### 12.5 Visualization API (Conceptual)

The engine should expose helper methods such as:
- `to_networkx_graph(game_state)`
- `draw_board(game_state, highlight_player=None)`

These functions:
- Read from GameState
- Never modify state
- May be used by:
  - Developers
  - Unit tests
  - RL rollout inspectors

### 12.6 Visualization in RL Workflows

Visualization is not part of the RL observation but is useful for:
- Inspecting failed episodes
- Debugging reward attribution
- Human-in-the-loop evaluation

All RL training uses structured state only.

---

## 13. Development Progression (Natural Order)

1. Lock board graph and adjacency
2. Define full state model
3. Implement phase and turn engine
4. Implement action resolution logic
5. Add visualization helpers
6. Add state validation and invariants
7. Implement RL interface and action masking
8. (Optional) Add UI layer

---

## 13. AI Delegation Strategy

**AI responsibilities:**
- Draft logic for individual rules
- Propose data structures
- Generate test scenarios
- Suggest RL encodings

**Human responsibilities:**
- Validate rule interpretations
- Approve state design
- Guide RL research direction

---

## 14. Definition of “Done”

- Full game playable headless
- No illegal states reachable
- RL agents can self-play complete games
- Game state fully reconstructible at any step

---

## 15. Next Steps

The next recommended design task is to **formalize the action enumeration tables** for:
- Choosing Actions phase
- Each Resolving Action context

This will finalize the RL interface and enable environment implementation.

