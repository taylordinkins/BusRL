"""Configuration constants for the Bus RL environment.

This module defines all configuration values for observation encoding,
action space sizing, and other RL-related constants.
"""

from dataclasses import dataclass
from typing import ClassVar

from core.constants import (
    MAX_PLAYERS,
    TOTAL_PASSENGERS,
    MAX_BUSES,
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
)


@dataclass(frozen=True)
class BoardConfig:
    """Configuration for the default Bus board topology.

    These values are derived from data/default_board.json and should
    match the actual board being used.
    """

    MAX_NODES: int = 36
    MAX_EDGES: int = 70
    MAX_BUILDING_SLOTS_PER_NODE: int = 2
    TRAIN_STATION_NODE_IDS: tuple[int, ...] = (8, 27)
    CENTRAL_PARK_NODE_IDS: tuple[int, ...] = (11, 14, 15, 20)


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for observation tensor dimensions.

    All dimension calculations are based on the board topology and
    game rules to create fixed-size tensors suitable for neural networks.
    """

    # Board dimensions
    MAX_NODES: int = BoardConfig.MAX_NODES
    MAX_EDGES: int = BoardConfig.MAX_EDGES
    MAX_BUILDING_SLOTS_PER_NODE: int = BoardConfig.MAX_BUILDING_SLOTS_PER_NODE

    # Game dimensions
    MAX_PLAYERS: int = MAX_PLAYERS
    MAX_PASSENGERS: int = TOTAL_PASSENGERS

    # Categorical dimensions
    ZONES: int = 4  # A, B, C, D
    BUILDING_TYPES: int = 4  # none, house, office, pub
    PHASES: int = 7  # All Phase enum values
    TIME_CLOCK_POSITIONS: int = 3  # house, office, pub

    # Action board dimensions
    ACTION_AREAS: int = 7
    MAX_SLOTS_PER_AREA: int = 6

    # Feature dimensions per component
    NODE_FEATURE_DIM: ClassVar[int] = 18
    EDGE_FEATURE_DIM: ClassVar[int] = 7
    PLAYER_FEATURE_DIM: ClassVar[int] = 10
    SLOT_FEATURE_DIM: ClassVar[int] = 4
    PASSENGER_FEATURE_DIM: ClassVar[int] = 5
    GLOBAL_FEATURE_DIM: ClassVar[int] = 27

    @property
    def node_features_size(self) -> int:
        """Total size of node features tensor."""
        return self.MAX_NODES * self.NODE_FEATURE_DIM

    @property
    def edge_features_size(self) -> int:
        """Total size of edge features tensor."""
        return self.MAX_EDGES * self.EDGE_FEATURE_DIM

    @property
    def player_features_size(self) -> int:
        """Total size of player features tensor."""
        return self.MAX_PLAYERS * self.PLAYER_FEATURE_DIM

    @property
    def action_board_size(self) -> int:
        """Total size of action board tensor."""
        return self.ACTION_AREAS * self.MAX_SLOTS_PER_AREA * self.SLOT_FEATURE_DIM

    @property
    def passenger_features_size(self) -> int:
        """Total size of passenger features tensor."""
        return self.MAX_PASSENGERS * self.PASSENGER_FEATURE_DIM

    @property
    def global_features_size(self) -> int:
        """Total size of global state tensor."""
        return self.GLOBAL_FEATURE_DIM

    @property
    def total_observation_dim(self) -> int:
        """Total dimension of the flat observation tensor."""
        return (
            self.node_features_size
            + self.edge_features_size
            + self.player_features_size
            + self.action_board_size
            + self.passenger_features_size
            + self.global_features_size
        )


@dataclass(frozen=True)
class ActionSpaceConfig:
    """Configuration for the discrete action space.

    Defines index ranges for each action category in the unified
    discrete action space.
    """

    # Board dimensions for action calculations
    MAX_NODES: int = BoardConfig.MAX_NODES
    MAX_EDGES: int = BoardConfig.MAX_EDGES
    MAX_BUILDING_SLOTS_PER_NODE: int = BoardConfig.MAX_BUILDING_SLOTS_PER_NODE
    MAX_PASSENGERS: int = TOTAL_PASSENGERS

    # Action categories and their sizes
    NUM_ACTION_AREAS: int = 7  # PLACE_MARKER destinations
    NUM_BUILDING_TYPES: int = 3  # house, office, pub (not none)

    # Computed action counts
    @property
    def place_marker_count(self) -> int:
        """Number of PLACE_MARKER actions (one per action area)."""
        return self.NUM_ACTION_AREAS

    @property
    def pass_count(self) -> int:
        """Number of PASS actions."""
        return 1

    @property
    def setup_building_count(self) -> int:
        """Number of SETUP_BUILDING actions: node x slot x building_type."""
        return self.MAX_NODES * self.MAX_BUILDING_SLOTS_PER_NODE * self.NUM_BUILDING_TYPES

    @property
    def setup_rail_count(self) -> int:
        """Number of SETUP_RAIL actions: one per edge."""
        return self.MAX_EDGES

    @property
    def line_expansion_count(self) -> int:
        """Number of LINE_EXPANSION actions: one per edge."""
        return self.MAX_EDGES

    @property
    def passengers_count(self) -> int:
        """Number of PASSENGERS distribution actions (0-5 to first station)."""
        return 6

    @property
    def buildings_count(self) -> int:
        """Number of BUILDINGS actions: node x slot x building_type."""
        return self.MAX_NODES * self.MAX_BUILDING_SLOTS_PER_NODE * self.NUM_BUILDING_TYPES

    @property
    def time_clock_count(self) -> int:
        """Number of TIME_CLOCK actions: advance or stop."""
        return 2

    @property
    def vrroomm_count(self) -> int:
        """Number of VRROOMM actions: passenger x node x slot."""
        return self.MAX_PASSENGERS * self.MAX_NODES * self.MAX_BUILDING_SLOTS_PER_NODE

    @property
    def skip_delivery_count(self) -> int:
        """Number of SKIP_DELIVERY actions."""
        return 1

    @property
    def noop_count(self) -> int:
        """Number of NOOP actions."""
        return 1

    # Index ranges (computed as properties)
    @property
    def place_marker_start(self) -> int:
        return 0

    @property
    def place_marker_end(self) -> int:
        return self.place_marker_start + self.place_marker_count

    @property
    def pass_idx(self) -> int:
        return self.place_marker_end

    @property
    def setup_building_start(self) -> int:
        return self.pass_idx + self.pass_count

    @property
    def setup_building_end(self) -> int:
        return self.setup_building_start + self.setup_building_count

    @property
    def setup_rail_start(self) -> int:
        return self.setup_building_end

    @property
    def setup_rail_end(self) -> int:
        return self.setup_rail_start + self.setup_rail_count

    @property
    def line_expansion_start(self) -> int:
        return self.setup_rail_end

    @property
    def line_expansion_end(self) -> int:
        return self.line_expansion_start + self.line_expansion_count

    @property
    def passengers_start(self) -> int:
        return self.line_expansion_end

    @property
    def passengers_end(self) -> int:
        return self.passengers_start + self.passengers_count

    @property
    def buildings_start(self) -> int:
        return self.passengers_end

    @property
    def buildings_end(self) -> int:
        return self.buildings_start + self.buildings_count

    @property
    def time_clock_start(self) -> int:
        return self.buildings_end

    @property
    def time_clock_end(self) -> int:
        return self.time_clock_start + self.time_clock_count

    @property
    def vrroomm_start(self) -> int:
        return self.time_clock_end

    @property
    def vrroomm_end(self) -> int:
        return self.vrroomm_start + self.vrroomm_count

    @property
    def skip_delivery_idx(self) -> int:
        return self.vrroomm_end

    @property
    def noop_idx(self) -> int:
        return self.skip_delivery_idx + self.skip_delivery_count

    @property
    def total_actions(self) -> int:
        """Total number of discrete actions in the unified action space."""
        return self.noop_idx + self.noop_count


@dataclass
class RewardConfig:
    """Configuration for reward calculation.

    Default values implement the sparse reward structure defined in
    the RL implementation plan.
    """

    # Delivery rewards
    delivery_reward: float = 1.0
    stolen_passenger_bonus: float = 0.1
    exclusive_delivery_bonus: float = 0.01

    # Network building rewards
    station_connection_reward: float = 0.1

    # Penalties
    time_stone_penalty: float = -0.01
    invalid_action_penalty: float = -1.0

    # Terminal rewards use point differential (no config needed)


# Default configuration instances
DEFAULT_BOARD_CONFIG = BoardConfig()
DEFAULT_OBS_CONFIG = ObservationConfig()
DEFAULT_ACTION_CONFIG = ActionSpaceConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
