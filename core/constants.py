"""Constants and enums for the Bus game engine."""

from enum import Enum


class Zone(Enum):
    """Building slot zones, ordered from innermost to outermost."""

    A = "A"  # Innermost
    B = "B"
    C = "C"
    D = "D"  # Outermost


class BuildingType(Enum):
    """Types of buildings that can be placed on the board."""

    HOUSE = "house"
    OFFICE = "office"
    PUB = "pub"


class Phase(Enum):
    """Game phases including setup and main game loop."""

    # Setup phases (executed once at game start)
    SETUP_BUILDINGS = "setup_buildings"
    SETUP_RAILS_FORWARD = "setup_rails_forward"
    SETUP_RAILS_REVERSE = "setup_rails_reverse"

    # Main game loop phases
    CHOOSING_ACTIONS = "choosing_actions"
    RESOLVING_ACTIONS = "resolving_actions"
    CLEANUP = "cleanup"
    GAME_OVER = "game_over"


class ActionAreaType(Enum):
    """Action areas on the action board."""

    LINE_EXPANSION = "line_expansion"
    BUSES = "buses"
    PASSENGERS = "passengers"
    BUILDINGS = "buildings"
    TIME_CLOCK = "time_clock"
    VRROOMM = "vrroomm"
    STARTING_PLAYER = "starting_player"


# Player limits
MIN_PLAYERS = 3
MAX_PLAYERS = 5

# Resource limits per player
TOTAL_ACTION_MARKERS = 20
TOTAL_RAIL_SEGMENTS = 25
MAX_BUSES = 5
INITIAL_BUSES = 1

# Game components
MAX_TIME_STONES = 5  # Maximum number of time stones (for >= 4 players)
TOTAL_PASSENGERS = 15  # Maximum number of passengers in the game
TOTAL_BUILDINGS_PER_TYPE = 30  # 90 total: 30 each of house, office, pub
INITIAL_PASSENGERS_AT_PARKS = 1  # 1 passenger per central park node at game start
MIN_MARKERS_PER_ROUND = 2  # Each player must place at least 2 markers per round

# Action area slot limits (max slots per area)
LINE_EXPANSION_SLOTS = 6  # F through A
BUSES_SLOTS = 1
PASSENGERS_SLOTS = 6
BUILDINGS_SLOTS = 6  # F through A
TIME_CLOCK_SLOTS = 1
VRROOMM_SLOTS = 6
STARTING_PLAYER_SLOTS = 1

# Action resolution order (fixed sequence for resolving action areas)
ACTION_RESOLUTION_ORDER = [
    ActionAreaType.LINE_EXPANSION,
    ActionAreaType.BUSES,
    ActionAreaType.PASSENGERS,
    ActionAreaType.BUILDINGS,
    ActionAreaType.TIME_CLOCK,
    ActionAreaType.VRROOMM,
    ActionAreaType.STARTING_PLAYER,
]

# Time clock order (building types in clockwise order, starts on HOUSE)
TIME_CLOCK_ORDER = [
    BuildingType.HOUSE,
    BuildingType.OFFICE,
    BuildingType.PUB,
]

# Zone priority for mandatory inner-first building placement
ZONE_PRIORITY = [Zone.A, Zone.B, Zone.C, Zone.D]
