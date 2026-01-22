# RL Implementation Plan for Bus Board Game

**Status:** ✅ Advanced Self-Play Training Complete — Multi-Policy & Elo System Implemented
**Goal:** Create a Gymnasium-compatible RL environment with PPO/MaskablePPO support for self-play training

---

## Executive Summary

This plan details the implementation of RL integration for the Bus board game engine. The design leverages the existing well-structured codebase with **minimal modifications** to core game logic.

**Key Design Decisions:**
- **Flat observation tensor** (1,458 floats) encoding all game state
- **Unified discrete action space** (1,670 actions) with action masking
- **Single-agent turn-based interface** with shared policy across all players
- **Self-relative player encoding** (current player always first)

---

## 1. Board Topology Summary

From `data/default_board.json`:
- **36 nodes** (IDs 0-35)
- **70 edges**
- **2 train stations** (nodes 8, 27)
- **4 central parks** (nodes 11, 14, 15, 20)
- **48 building slots** across 4 zones (A: 16 inner, B: 12, C: 8, D: 12 outer)

---

## 2. Files Created

```
rl/
    __init__.py          # ✅ Updated with all exports
    config.py            # ✅ Configuration constants
    observation.py       # ✅ ObservationEncoder class
    action_space.py      # ✅ ActionSpace, ActionMapping classes
    action_masking.py    # ✅ ActionMaskGenerator class
    reward.py            # ✅ RewardConfig, RewardCalculator classes
    bus_env.py           # ✅ BusEnv (Gymnasium environment)
    wrappers.py          # ✅ Self-play and training utilities
    opponent_pool.py     # ✅ OpponentPool, CheckpointInfo, PoolConfig (integrated with EloTracker)
    callbacks.py         # ✅ OpponentPoolCallback, OpponentPoolEvalCallback, MultiPolicyTrainingCallback
    elo_tracker.py       # ✅ EloTracker, MatchResult, HeadToHeadStats
    multi_policy_env.py  # ✅ MultiPolicyBusEnv, PolicySlot, MatchRunner
    mcts.py              # ✅ MCTS, MCTSNode, MCTSConfig (Monte Carlo Tree Search)
    mcts_player.py       # ✅ MCTSPlayer, PolicyPlayer, MCTSPlayerStats
scripts/
    train.py             # ✅ Training script with MaskablePPO (--use-opponent-pool, --multi-policy flags)
    evaluate.py          # ✅ Evaluation script with MCTS support (--use-mcts, --compare-mcts flags)
tests/
    test_opponent_pool.py # ✅ Unit tests for opponent pool

---

## 3. Observation Encoding (`rl/observation.py`)

### 3.1 Configuration Constants

```python
@dataclass
class ObservationConfig:
    MAX_NODES: int = 36
    MAX_EDGES: int = 70
    MAX_BUILDING_SLOTS_PER_NODE: int = 2
    MAX_PLAYERS: int = 5
    MAX_PASSENGERS: int = 15
    ACTION_AREAS: int = 7
    MAX_SLOTS_PER_AREA: int = 6
    ZONES: int = 4           # A, B, C, D
    BUILDING_TYPES: int = 4  # none, house, office, pub
    PHASES: int = 7
```

### 3.2 Observation Components

| Component | Shape | Features | Total Floats |
|-----------|-------|----------|--------------|
| Node features | [36, 18] | position, flags, 2 slots (zone+building+occupied) | 648 |
| Edge features | [70, 7] | endpoints, is_empty, player_rails[5] | 490 |
| Player features | [5, 10] | markers, rails, buses, score, stones, passed | 50 |
| Action board | [7, 6, 4] | slot occupied, player, is_current, order | 168 |
| Passenger features | [15, 5] | exists, location, station/park/matching flags | 75 |
| Global state | [27] | phase, round, clock, resolution progress | 27 |
| **Total** | | | **~1,458** |

### 3.3 Self-Relative Player Encoding

Players are reordered so current player is always index 0:
```python
# If current player is 2 in a 4-player game:
# Original order: [P0, P1, P2, P3]
# Encoded order:  [P2, P3, P0, P1]  (current player first, then clockwise)
```

### 3.4 Key Method

```python
class ObservationEncoder:
    def encode(self, state: GameState, current_player_id: int) -> np.ndarray:
        """Encode complete game state into flat observation tensor."""
```

---

## 4. Action Space Design (`rl/action_space.py`)

### 4.1 Unified Discrete Action Space

All actions across all phases mapped to single discrete space:

| Action Category | Index Range | Count | Description |
|-----------------|-------------|-------|-------------|
| PLACE_MARKER | 0-6 | 7 | One per action area |
| PASS | 7 | 1 | Pass during choosing phase |
| SETUP_BUILDING | 8-223 | 216 | node(36) x slot(2) x type(3) |
| SETUP_RAIL | 224-293 | 70 | One per edge |
| LINE_EXPANSION | 294-363 | 70 | One per edge |
| PASSENGERS | 364-369 | 6 | Distribution to station 0 (0-5) |
| BUILDINGS | 370-585 | 216 | node(36) x slot(2) x type(3) |
| TIME_CLOCK | 586-587 | 2 | Advance or Stop |
| VRROOMM | 588-1667 | 1080 | passenger(15) x node(36) x slot(2) |
| SKIP_DELIVERY | 1668 | 1 | End Vrroomm early (no more deliveries) |
| NOOP | 1669 | 1 | Auto-advance phases |
| **Total** | | **1,670** | |

**Note:** Vrroomm uses single-delivery-per-step approach. Each step the agent picks one passenger+destination, repeating until all buses used or agent chooses SKIP_DELIVERY.

### 4.2 Key Classes

```python
class ActionMapping:
    """Bidirectional mapping between flat indices and Action objects."""

    def index_to_action(self, idx: int, state: GameState) -> Action
    def action_to_index(self, action: Action) -> int
    def get_action_range(self, action_type: ActionType) -> tuple[int, int]
```

---

## 5. Action Masking (`rl/action_masking.py`)

### 5.1 Mask Generation

```python
class ActionMaskGenerator:
    def generate_mask(
        self,
        state: GameState,
        valid_actions: list[Action]
    ) -> np.ndarray:
        """Return boolean mask of shape [TOTAL_ACTIONS]."""
        mask = np.zeros(TOTAL_ACTIONS, dtype=np.bool_)
        for action in valid_actions:
            idx = self.action_mapping.action_to_index(action)
            mask[idx] = True
        return mask
```

### 5.2 Integration with MaskablePPO

The environment exposes `action_masks()` method compatible with `sb3_contrib.MaskablePPO`:
```python
def action_masks(self) -> np.ndarray:
    valid_actions = self._get_valid_actions()
    return self._mask_generator.generate_mask(self.state, valid_actions)
```

---

## 6. Reward Function (`rl/reward.py`)

### 6.1 Sparse Rewards (Default)

| Event | Reward | Notes |
|-------|--------|-------|
| **Passenger delivered** | **+1.0** | Per passenger delivered during Vrroomm |
| Stolen passenger bonus | +0.1 | Additional reward if passenger was on another player's line |
| Exclusive delivery bonus | +0.01 | Additional if destination node not touching opponent's line |
| Train station connection | +0.1 | First time player's rail network reaches each train station |
| Time stone taken | -0.01 | Penalty when stopping clock |
| **Terminal (1st place)** | **+Δ score** | Point differential over 2nd place player |
| **Terminal (others)** | **-Δ score** | Negative point differential vs 1st place |

### 6.2 Reward Logic Details

**Delivery Rewards:**
```python
# Base delivery reward
reward += 1.0

# Bonus: passenger was on opponent's rail network before pickup
if passenger_was_on_opponent_line:
    reward += 0.1

# Bonus: destination node has no opponent rails touching it
if destination_node_not_on_opponent_network:
    reward += 0.01
```

**Train Station Connection:**
- Track which stations each player has connected to
- First connection to station 8: +0.1
- First connection to station 27: +0.1
- Only awarded once per player per station

**Terminal Rewards:**
```python
final_scores = sorted([(p.get_final_score(), p.player_id) for p in players], reverse=True)
first_place_score = final_scores[0][0]
second_place_score = final_scores[1][0] if len(final_scores) > 1 else 0

if player_id == first_place_player:
    terminal_reward = first_place_score - second_place_score  # Positive
else:
    terminal_reward = player_score - first_place_score  # Negative
```

### 6.3 Key Class

```python
@dataclass
class RewardConfig:
    delivery_reward: float = 1.0
    stolen_passenger_bonus: float = 0.1
    exclusive_delivery_bonus: float = 0.01
    station_connection_reward: float = 0.1
    time_stone_penalty: float = -0.01
    # Terminal uses point differential (no config needed)

class RewardCalculator:
    def __init__(self, config: RewardConfig):
        self._stations_connected: dict[int, set[int]] = {}  # player_id -> set of station node_ids

    def compute_reward(
        self,
        state: GameState,
        prev_state: GameState,
        current_player_id: int,
        done: bool
    ) -> float

    def _check_station_connections(self, state: GameState, player_id: int) -> float:
        """Award +0.1 for each newly connected train station."""

    def _compute_delivery_bonuses(
        self,
        passenger_id: int,
        from_node: int,
        to_node: int,
        delivering_player: int,
        state: GameState
    ) -> float:
        """Compute stolen passenger and exclusive delivery bonuses."""

    def _compute_terminal_reward(self, state: GameState, player_id: int) -> float:
        """Compute point differential terminal reward."""
```

---

## 7. Gymnasium Environment (`rl/bus_env.py`)

### 7.1 Environment Class

```python
class BusEnv(gym.Env):
    """Gymnasium environment for Bus board game.

    - Single-agent turn-based interface
    - All players use shared policy
    - Observations from current player's perspective
    - Action masking for legal move enforcement
    """

    observation_space = spaces.Box(low=-1, high=1, shape=(1458,), dtype=np.float32)
    action_space = spaces.Discrete(1670)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]
    def action_masks(self) -> np.ndarray
    def render(self) -> Optional[str]
```

### 7.2 Step Logic

1. Convert flat action index to `Action` object
2. Execute action via `GameEngine.step()` or `ActionResolver`
3. Auto-advance through phases with no player decisions (BUSES, STARTING_PLAYER, CLEANUP)
4. Compute reward for acting player
5. Return observation from new current player's perspective

### 7.3 Info Dictionary

```python
info = {
    "phase": str,
    "round": int,
    "current_player": int,
    "action_mask": np.ndarray,
    "valid_action_count": int,
    "scores": dict[int, int],
}
```

---

## 8. Modifications to Existing Code

### 8.1 Required Changes (Minimal)

**`engine/game_engine.py`** - Add resolution phase access (✓ Implemented):
```python
def get_or_create_action_resolver(self) -> "ActionResolver":
    """Get or create an ActionResolver for the current resolution phase."""
    from .action_resolver import ActionResolver
    return ActionResolver(self.state)

def get_valid_resolution_actions_for_rl(self) -> list[dict]:
    """Get valid resolution actions in dict format for RL.

    Returns list of action dicts during RESOLVING_ACTIONS phase,
    empty list otherwise.
    """
    if self.state.phase != Phase.RESOLVING_ACTIONS:
        return []
    from .action_resolver import ActionResolver, ResolutionStatus
    resolver = ActionResolver(self.state)
    resolver.start_resolution()
    context = resolver.get_context()
    if context.status == ResolutionStatus.AWAITING_INPUT:
        return context.valid_actions
    return []
```

**`core/game_state.py`** - Add helper method:
```python
def get_player(self, player_id: int) -> Player:
    """Get player by ID (convenience method)."""
    return self.players[player_id]
```
**Note:** This method already exists in the codebase - no changes needed.

### 8.2 No Changes Required

- `core/board.py` - Already has all needed methods
- `core/player.py` - Complete resource tracking
- `core/action_board.py` - Full marker management
- `engine/action_resolver.py` - Complete resolution logic
- All resolvers - Working correctly

---

## 9. Implementation Order

### Phase 1: Core RL Infrastructure ✓ Complete
1. [x] `rl/config.py` - Constants and configuration
2. [x] `rl/observation.py` - State encoding
3. [x] `rl/action_space.py` - Action mapping

### Phase 2: Environment Core ✓ Complete
4. [x] `rl/action_masking.py` - Mask generation
5. [x] `rl/reward.py` - Reward calculation
6. [x] `rl/bus_env.py` - Main environment

### Phase 3: Integration & Testing ✓ Complete
7. [x] Minor engine modifications - Added `get_or_create_action_resolver()` and `get_valid_resolution_actions_for_rl()` to `engine/game_engine.py`
8. [x] Unit tests for all RL modules (`tests/test_rl_*.py`)
9. [x] Integration test: random agent plays full games (`tests/test_rl_integration.py`)

### Phase 4: Training Infrastructure ✓ Complete
10. [x] `rl/wrappers.py` - Self-play, vectorized env utilities
11. [x] Training script with MaskablePPO (`scripts/train.py`)
12. [x] Evaluation script (`scripts/evaluate.py`)

### Phase 5: Advanced Self-Play Training ✓ Complete
13. [x] Opponent pool for self-play diversity (see Section 13)
14. [x] Elo rating system for checkpoint evaluation (`rl/elo_tracker.py`)
15. [x] Prioritized Fictitious Self-Play (PFSP) matchmaking (`rl/multi_policy_env.py`)

### Phase 6: Human Integration (Future)
16. [ ] Human vs AI play in GUI - load saved model and interface with GUI through agent's policy
17. [ ] Model selection UI for choosing opponent strength

---

## 10. Verification Plan

### 10.1 Unit Tests
- Observation encoder produces correct shapes
- Action mapping is bijective (index <-> action)
- Mask generation matches valid actions
- Reward computation is correct

### 10.2 Integration Tests (✓ Implemented in `tests/test_rl_integration.py`)
- [x] Random agent completes full games without errors (3, 4, 5 players)
- [x] All phases transition correctly
- [x] Action masking prevents illegal moves
- [x] Multiple games can be played on same environment
- [x] Environment cloning works correctly
- [x] Observations always within [0, 1] bounds
- [x] Rewards are always finite
- [x] Info dictionary contains required keys

### 10.3 RL Smoke Test
- MaskablePPO trains for 1000 steps without crashes
- Loss decreases over training
- Agent beats random baseline after training

---

## 11. Dependencies

```
# requirements.txt additions
gymnasium>=0.29.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0  # For MaskablePPO
numpy>=1.24.0
torch>=2.0.0
```

---

## 12. Current Implementation Summary

### What's Working ✅

| Component | File | Status |
|-----------|------|--------|
| Environment | `rl/bus_env.py` | Complete - Gymnasium-compatible, action masking, auto-advancement |
| Observations | `rl/observation.py` | Complete - Self-relative encoding, ~1,458 features |
| Actions | `rl/action_space.py` | Complete - 1,670 discrete actions, bidirectional mapping |
| Masking | `rl/action_masking.py` | Complete - MaskablePPO compatible |
| Rewards | `rl/reward.py` | Complete - Sparse + shaping, per-player tracking |
| Wrappers | `rl/wrappers.py` | Basic - Per-player stats tracking |
| Training | `scripts/train.py` | Complete - MaskablePPO with checkpoints |
| Evaluation | `scripts/evaluate.py` | Basic - Self-play evaluation only |

### Current Training Setup

The implementation now supports two training modes:

**1. Basic Self-Play** (default):
- Single policy controls all players
- All players share the same weights during training
- Checkpoints saved to opponent pool but not used as opponents
- Enable with: `python scripts/train.py`

**2. Multi-Policy Self-Play with PFSP** (recommended):
- Training policy plays against diverse opponents from checkpoint pool
- Opponents sampled using PFSP (Prioritized Fictitious Self-Play)
- Elo ratings track checkpoint performance
- Head-to-head evaluation updates win rates
- Enable with: `python scripts/train.py --use-opponent-pool --multi-policy`

Multi-policy training helps prevent:
- **Policy collapse** - diverse opponents maintain robustness
- **Cyclic behavior** - PFSP targets challenging opponents
- **Poor generalization** - training against historical strategies

---

## 13. Self-Play Training Best Practices

This section outlines the recommended approach for robust self-play training, based on techniques from AlphaStar, OpenAI Five, and other successful game AI systems.

### 13.1 The Opponent Pool Architecture

**Purpose:** Maintain diverse opponents to prevent policy collapse and encourage robust strategies.

#### Core Components

```
rl/
    opponent_pool.py      # Checkpoint management & sampling
    matchmaking.py        # PFSP or league-based matchmaking
    multi_policy_env.py   # Environment with different policies per player
    elo_tracker.py        # Rating system for checkpoints
scripts/
    train_selfplay.py     # Training loop with opponent sampling
```

#### Opponent Pool Design

```python
class OpponentPool:
    """Manages a pool of past policy checkpoints for diverse training."""

    def __init__(
        self,
        pool_size: int = 20,           # Max checkpoints to retain
        save_interval: int = 50_000,   # Steps between checkpoint saves
        min_elo_gap: int = 100,        # Minimum Elo spread to keep
    ):
        self.checkpoints: list[CheckpointInfo] = []
        self.current_policy: MaskablePPO = None

    def save_checkpoint(self, model: MaskablePPO, step: int, elo: float):
        """Save current policy as a new checkpoint."""

    def sample_opponent(self, method: str = "pfsp") -> MaskablePPO:
        """Sample an opponent from the pool.

        Methods:
        - "uniform": Equal probability for all checkpoints
        - "latest": Bias toward recent checkpoints
        - "pfsp": Prioritized Fictitious Self-Play (recommended)
        """

    def prune_pool(self):
        """Remove redundant checkpoints (similar Elo, old, rarely sampled)."""
```

### 13.2 Prioritized Fictitious Self-Play (PFSP)

**Best Practice:** Sample opponents based on win rate against them.

PFSP prioritizes opponents where the current policy has ~50% win rate, as these provide the most learning signal.

```python
def pfsp_sampling_weight(win_rate: float) -> float:
    """Compute sampling priority based on win rate.

    Opponents with ~50% win rate are most valuable for learning.
    Easy (>70%) and hard (<30%) opponents provide less signal.
    """
    # f(x) = x(1-x) peaks at 0.5
    return win_rate * (1 - win_rate) + 0.1  # Small epsilon for exploration
```

#### Matchmaking Implementation

```python
class PFSPMatchmaker:
    """Prioritized Fictitious Self-Play matchmaking."""

    def __init__(self, opponent_pool: OpponentPool):
        self.pool = opponent_pool
        self.win_rates: dict[str, float] = {}  # checkpoint_id -> win rate

    def select_opponents(self, num_players: int = 4) -> list[Policy]:
        """Select opponents for a game.

        Returns list of policies for each player slot.
        Slot 0 is always the training policy.
        """
        opponents = [self.pool.current_policy]  # Slot 0: training agent

        for _ in range(num_players - 1):
            # 80% PFSP sampling, 20% current policy (self-play)
            if random.random() < 0.8:
                checkpoint = self._sample_by_pfsp()
                opponents.append(self._load_checkpoint(checkpoint))
            else:
                opponents.append(self.pool.current_policy)

        random.shuffle(opponents[1:])  # Randomize opponent positions
        return opponents
```

### 13.3 Multi-Policy Environment Wrapper

To train against diverse opponents, the environment needs to support different policies per player slot.

```python
class MultiPolicyBusEnv(gym.Wrapper):
    """Environment that supports different policies for different players.

    During training:
    - Player 0 (or the training slot) uses the current training policy
    - Other players use sampled opponents from the pool
    - Only returns experiences for the training player
    """

    def __init__(
        self,
        env: BusEnv,
        matchmaker: PFSPMatchmaker,
        training_player_slot: int = 0,
    ):
        super().__init__(env)
        self.matchmaker = matchmaker
        self.training_slot = training_player_slot
        self.opponent_policies: list[Policy] = []

    def reset(self, **kwargs):
        """Reset and assign new opponents for this episode."""
        obs, info = self.env.reset(**kwargs)

        # Sample opponents for this episode
        self.opponent_policies = self.matchmaker.select_opponents(
            self.env.num_players
        )

        # If not training player's turn, let opponents play
        obs = self._advance_to_training_player(obs)
        return obs, info

    def step(self, action: int):
        """Execute action and handle opponent turns."""
        obs, reward, term, trunc, info = self.env.step(action)

        # Let opponents play until it's training player's turn again
        while not (term or trunc):
            current_player = self.env.get_current_player()
            if current_player == self.training_slot:
                break

            # Opponent's turn - use their policy
            opp_policy = self.opponent_policies[current_player]
            opp_action = self._get_opponent_action(opp_policy, obs)
            obs, _, term, trunc, info = self.env.step(opp_action)

        return obs, reward, term, trunc, info
```

### 13.4 Elo Rating System

Track relative strength of checkpoints to enable intelligent matchmaking and measure training progress.

```python
class EloTracker:
    """Track Elo ratings for all checkpoints in the pool."""

    def __init__(self, k_factor: float = 32.0, initial_elo: float = 1500.0):
        self.ratings: dict[str, float] = {}
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.match_history: list[MatchResult] = []

    def update_ratings(self, winner_id: str, loser_id: str):
        """Update Elo ratings after a match."""
        r_winner = self.ratings.get(winner_id, self.initial_elo)
        r_loser = self.ratings.get(loser_id, self.initial_elo)

        # Expected scores
        e_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
        e_loser = 1 - e_winner

        # Update ratings
        self.ratings[winner_id] = r_winner + self.k_factor * (1 - e_winner)
        self.ratings[loser_id] = r_loser + self.k_factor * (0 - e_loser)

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Return checkpoints sorted by Elo rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 13.5 Training Loop with Opponent Pool

```python
def train_with_opponent_pool(args):
    """Training loop with opponent pool and PFSP matchmaking."""

    # Initialize components
    pool = OpponentPool(pool_size=20, save_interval=50_000)
    elo_tracker = EloTracker()
    matchmaker = PFSPMatchmaker(pool, elo_tracker)

    # Create environment with multi-policy support
    def make_env():
        env = BusEnv(num_players=args.num_players)
        env = MultiPolicyBusEnv(env, matchmaker)
        return env

    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    # Initialize model
    model = MaskablePPO("MlpPolicy", vec_env, ...)
    pool.current_policy = model

    # Save initial checkpoint
    pool.save_checkpoint(model, step=0, elo=1500.0)

    # Training loop
    total_steps = 0
    while total_steps < args.total_timesteps:
        # Train for one iteration
        model.learn(total_timesteps=args.steps_per_iteration, reset_num_timesteps=False)
        total_steps += args.steps_per_iteration

        # Periodically save checkpoints
        if total_steps % args.checkpoint_interval == 0:
            # Evaluate current policy against pool
            current_elo = evaluate_against_pool(model, pool, n_games=50)
            pool.save_checkpoint(model, total_steps, current_elo)

            # Update win rates for PFSP
            matchmaker.update_win_rates(model, pool.checkpoints)

            # Prune pool if needed
            pool.prune_pool()

        # Log metrics
        log_training_metrics(model, pool, elo_tracker)
```

### 13.6 Recommended Hyperparameters for Self-Play

| Parameter | Naive Self-Play | With Opponent Pool |
|-----------|-----------------|-------------------|
| Learning Rate | 3e-4 | 1e-4 (more stable) |
| Entropy Coefficient | 0.01 | 0.02-0.05 (encourage exploration) |
| n_steps | 2048 | 4096 (longer rollouts) |
| Checkpoint Interval | 50k | 25k-50k |
| Pool Size | N/A | 15-30 |
| PFSP Self-Play Ratio | N/A | 0.2 (20% current, 80% pool) |
| Eval Games per Checkpoint | N/A | 50-100 |

### 13.7 Evaluation Strategy

```python
def comprehensive_evaluation(model: MaskablePPO, pool: OpponentPool) -> dict:
    """Evaluate model against various benchmarks."""

    results = {}

    # 1. Self-play win rate (should be ~50%)
    results["self_play_win_rate"] = eval_vs_self(model, n_games=100)

    # 2. Win rate vs random agent (should be >90% for trained model)
    results["vs_random"] = eval_vs_random(model, n_games=100)

    # 3. Win rate vs each checkpoint tier
    for tier in ["top_3", "middle", "oldest"]:
        checkpoints = pool.get_tier(tier)
        results[f"vs_{tier}"] = eval_vs_checkpoints(model, checkpoints, n_games=50)

    # 4. Average Elo rating
    results["elo"] = compute_elo_vs_pool(model, pool, n_games=100)

    # 5. Strategy diversity metrics
    results["action_entropy"] = measure_action_diversity(model)

    return results
```

### 13.8 Implementation Roadmap

#### Step 1: Basic Opponent Pool ✅ Complete
- [x] `OpponentPool` class with checkpoint save/load (`rl/opponent_pool.py`)
- [x] Uniform sampling from pool (also: latest, pfsp, elo_weighted methods)
- [x] Integration with existing training script (`scripts/train.py --use-opponent-pool`)
- [x] `OpponentPoolCallback` for automatic checkpoint saving (`rl/callbacks.py`)
- [x] `CheckpointInfo` and `PoolConfig` dataclasses
- [x] Pool state persistence (JSON serialization)
- [x] Pruning strategies: oldest, lowest_elo, least_diverse
- [x] Unit tests (`tests/test_opponent_pool.py`)

#### Step 2: Multi-Policy Environment ✅ Complete
- [x] `MultiPolicyBusEnv` wrapper (`rl/multi_policy_env.py`)
- [x] `PolicySlot` class for managing policy assignments per player
- [x] Opponent action selection during non-training turns
- [x] Experience collection only for training player
- [x] `MatchRunner` utility for running evaluation matches
- [x] Integration with training script (`scripts/train.py --multi-policy`)

#### Step 3: Elo Rating System ✅ Complete
- [x] `EloTracker` class (`rl/elo_tracker.py`)
- [x] Standard Elo rating updates (two-player)
- [x] Multi-player Elo rating updates
- [x] Head-to-head statistics tracking (`HeadToHeadStats`)
- [x] Match history recording (`MatchResult`)
- [x] Leaderboard retrieval
- [x] State persistence (JSON)
- [x] Integration with `OpponentPool`

#### Step 4: PFSP Matchmaking ✅ Complete
- [x] Win rate tracking per checkpoint (via `HeadToHeadStats`)
- [x] PFSP sampling weights in `OpponentPool.sample_opponent(method="pfsp")`
- [x] Dynamic matchmaking during training via `MultiPolicyBusEnv`
- [x] `OpponentPoolEvalCallback` for actual head-to-head evaluation

#### Step 5: Training Script Integration ✅ Complete
- [x] Updated `scripts/train.py` with full multi-policy support
- [x] TensorBoard logging for Elo, win rates, pool diversity
- [x] `MultiPolicyTrainingCallback` for Elo sync during training
- [x] New CLI flags: `--multi-policy`, `--self-play-prob`, `--sampling-method`
- [x] Pool evaluation flags: `--pool-eval-interval`, `--pool-eval-games`
- [x] Elo configuration: `--elo-k-factor`

### 13.9 Usage Examples

**Basic training (naive self-play):**
```bash
python scripts/train.py --total-timesteps 1000000
```

**Training with opponent pool (checkpoint diversity):**
```bash
python scripts/train.py --use-opponent-pool --pool-size 20 --pool-save-interval 50000
```

**Full multi-policy training with PFSP (recommended):**
```bash
python scripts/train.py \
    --use-opponent-pool \
    --multi-policy \
    --pool-size 20 \
    --sampling-method pfsp \
    --self-play-prob 0.2 \
    --pool-eval-interval 100000 \
    --total-timesteps 5000000
```

**Key flags:**
- `--use-opponent-pool`: Enable checkpoint saving and pool management
- `--multi-policy`: Train against pool opponents instead of pure self-play
- `--sampling-method`: How to select opponents (uniform, latest, pfsp, elo_weighted)
- `--self-play-prob`: Probability of playing against current policy vs pool opponent
- `--pool-eval-interval`: Steps between head-to-head evaluation (updates Elo ratings)
- `--prune-strategy`: How to remove old checkpoints (oldest, lowest_elo, least_diverse)

### 13.10 Monitoring & Debugging

Key metrics to track during training:

```python
# TensorBoard logging
writer.add_scalar("elo/current_policy", current_elo, step)
writer.add_scalar("elo/best_checkpoint", pool.best_elo(), step)
writer.add_scalar("pool/size", len(pool.checkpoints), step)
writer.add_scalar("pool/elo_spread", pool.elo_spread(), step)
writer.add_scalar("matchmaking/avg_win_rate_vs_opponents", avg_wr, step)
writer.add_scalar("eval/vs_random_win_rate", vs_random, step)
writer.add_histogram("pfsp/sampling_weights", weights, step)
```

**Warning Signs:**
- Elo not increasing after many iterations → Policy may be stuck
- Win rate vs random decreasing → Policy collapse
- All win rates near 50% → Possible cyclic behavior, increase pool diversity
- Elo spread shrinking → Pool not diverse enough, lower pruning threshold

---

## 14. MCTS Integration ✅ Complete

Monte Carlo Tree Search enhances the trained policy at inference time by performing lookahead search.

### 14.1 Implementation

**New Files:**
- `rl/mcts.py` - Core MCTS implementation with PUCT-based tree search
- `rl/mcts_player.py` - MCTSPlayer wrapper for easy integration

**Key Classes:**
- `MCTSConfig` - Configuration dataclass for MCTS parameters
- `MCTSNode` - Tree node with visit counts, values, and policy priors
- `MCTS` - Main search class using policy network for priors and value estimation
- `MCTSPlayer` - Player wrapper for evaluation and gameplay
- `PolicyPlayer` - Baseline player using direct policy inference

### 14.2 How It Works

1. **Tree Building:** Start from current state, expand using cloned environments
2. **Selection:** PUCT formula balances exploration (prior) and exploitation (Q-value)
3. **Expansion:** Use policy network to get action priors, mask illegal actions
4. **Evaluation:** Use value network for leaf evaluation (or random rollout)
5. **Backpropagation:** Update visit counts and values up to root
6. **Action Selection:** Choose action based on visit counts with temperature

### 14.3 Usage

**Basic MCTS evaluation:**
```bash
python scripts/evaluate.py path/to/model.zip --use-mcts --mcts-simulations 100
```

**Full configuration:**
```bash
python scripts/evaluate.py path/to/model.zip \
    --use-mcts \
    --mcts-simulations 200 \
    --mcts-c-puct 1.5 \
    --mcts-temperature 0.1 \
    --mcts-dirichlet-epsilon 0.25
```

**Compare MCTS vs policy-only:**
```bash
python scripts/evaluate.py path/to/model.zip --compare-mcts --num-games 20
```

### 14.4 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_simulations` | 100 | MCTS simulations per move |
| `c_puct` | 1.5 | Exploration constant (higher = more exploration) |
| `temperature` | 1.0 | Action selection temperature (0 = greedy) |
| `use_value_network` | True | Use value head vs random rollout |
| `dirichlet_alpha` | 0.3 | Dirichlet noise alpha for root |
| `dirichlet_epsilon` | 0.25 | Weight of Dirichlet noise (0 = disabled) |

### 14.5 Performance Considerations

- **Inference time:** ~100 simulations adds ~1-2 seconds per move
- **Memory:** Each node stores a cloned environment
- **GPU utilization:** Policy/value queries are batched when possible
- **Scaling:** More simulations generally improve play quality

---

## 15. Future Extensions

### 15.1 Gumbel AlphaZero / Gumbel MuZero (Proposed)

**Concept:** Replace UCB selection with Gumbel-Top-k sampling for more sample-efficient search.

**Key Differences from Standard MCTS:**
1. Use Gumbel noise instead of UCB for action selection
2. Sequential halving to select top-k actions
3. Policy improvement via completed Q-values
4. Better theoretical guarantees for discrete action spaces

**Benefits:**
- More sample-efficient than traditional MCTS
- Works well with fewer simulations
- Used in recent DeepMind work (MuZero Unplugged)

**Implementation Notes:**
- Would add `gumbel` option to `MCTSConfig.selection_method`
- Requires modified selection phase in `MCTS.search()`
- Reference: "Policy Improvement by Planning with Gumbel" (Danihelka et al., 2022)

### 15.2 Other Future Work

- **Graph Neural Networks** - GNN-based policy for board topology
- **Curriculum Learning** - Simplified boards, fewer players, reduced complexity
- **Population-Based Training (PBT)** - Hyperparameter evolution during training
- **Model Distillation** - Compress large models for faster inference
- **Human Benchmark Games** - Curated game logs for evaluation
- **AlphaZero-style Training** - Use MCTS during training (not just inference)

---

## 16. Technical Notes & Best Practices

### 16.1 MCTS Reward Scaling and Alignment
Previously, MCTS used a raw score differential normalized by the max score for terminal evaluation. This created a scale mismatch with the trained Value Network, which predicts raw score differentials (reward sums).
- **Current Implementation:** MCTS now uses the `RewardCalculator` directly to compute terminal values. This ensures that the search's "Value" (Q) is in the same units and scale as the rewards the agent sees during training.
- **Future Suggestion:** If the score differentials become very large (e.g., > 20 points), consider applying a symmetric log transformation or hyperbolic tangent to stabilize the value estimates in both training and search.

### 16.2 MCTS Environment Cloning Performance
Currently, `MCTS` calls `env.clone()` during the expansion of every node. 
- **Current State:** This is robust but computationally expensive because `GameState.clone()` performs a deep copy of many nested objects.
- **Optimization Strategy:** To scale to 1000+ simulations/second, we should implement a "Lightweight Reset/Undo" or "State Rollback" mechanism in the `GameEngine`. Instead of cloning the entire board, we could store a stack of `Action` objects and their inverse results to backtrack through the tree.

### 16.3 Multi-Policy Terminal Rewards
In environments with multiple policies (`MultiPolicyBusEnv`), a critical logical edge case exists where the game ends on an opponent's turn.
- **Problem:** Standard `step()` only returns rewards for the player who just acted. If an opponent delivers the last passenger, the training agent would normally receive 0 reward for that step, missing its terminal win/loss signal.
- **Fix:** the `MultiPolicyBusEnv` now explicitly checks if the game ended during `_handle_opponent_turns`. If so, it manually computes the terminal reward for the training player and adds it to the `cumulative_reward`.

### 16.4 Entropy Decay in Training
To balance exploration and exploitation, we've introduced an `EntropyDecayCallback`.
- **Strategy:** Start with a higher entropy coefficient (e.g., 0.05) to explore the vast action space of *Bus*, and linearly decay it to a lower value (e.g., 0.01) as training progresses. This prevents the policy from becoming deterministic too early, which is vital in self-play to discover counter-strategies.
