"""Core data models for the Bus game engine."""

from .constants import (
    Zone,
    BuildingType,
    Phase,
    ActionAreaType,
    MIN_PLAYERS,
    MAX_PLAYERS,
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
    MAX_BUSES,
    INITIAL_BUSES,
    TOTAL_BUILDINGS_PER_TYPE,
    INITIAL_PASSENGERS_AT_PARKS,
    MIN_MARKERS_PER_ROUND,
    LINE_EXPANSION_SLOTS,
    BUSES_SLOTS,
    PASSENGERS_SLOTS,
    BUILDINGS_SLOTS,
    TIME_CLOCK_SLOTS,
    VRROOMM_SLOTS,
    STARTING_PLAYER_SLOTS,
    ACTION_RESOLUTION_ORDER,
    TIME_CLOCK_ORDER,
    ZONE_PRIORITY,
)

from .board import (
    NodeId,
    EdgeId,
    make_edge_id,
    BuildingSlot,
    NodeState,
    RailSegment,
    EdgeState,
    BoardGraph,
)

from .player import Player

from .components import Passenger, PassengerManager

from .action_board import ActionSlot, ActionArea, ActionBoard

from .game_state import GlobalState, GameState

__all__ = [
    # Constants
    "Zone",
    "BuildingType",
    "Phase",
    "ActionAreaType",
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "TOTAL_ACTION_MARKERS",
    "TOTAL_RAIL_SEGMENTS",
    "MAX_BUSES",
    "INITIAL_BUSES",
    "TOTAL_BUILDINGS_PER_TYPE",
    "INITIAL_PASSENGERS_AT_PARKS",
    "MIN_MARKERS_PER_ROUND",
    "LINE_EXPANSION_SLOTS",
    "BUSES_SLOTS",
    "PASSENGERS_SLOTS",
    "BUILDINGS_SLOTS",
    "TIME_CLOCK_SLOTS",
    "VRROOMM_SLOTS",
    "STARTING_PLAYER_SLOTS",
    "ACTION_RESOLUTION_ORDER",
    "TIME_CLOCK_ORDER",
    "ZONE_PRIORITY",
    # Board
    "NodeId",
    "EdgeId",
    "make_edge_id",
    "BuildingSlot",
    "NodeState",
    "RailSegment",
    "EdgeState",
    "BoardGraph",
    # Player
    "Player",
    # Components
    "Passenger",
    "PassengerManager",
    # Action Board
    "ActionSlot",
    "ActionArea",
    "ActionBoard",
    # Game State
    "GlobalState",
    "GameState",
]
