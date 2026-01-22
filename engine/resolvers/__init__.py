"""Action resolvers for the Bus game engine.

This module contains the resolvers for each action area on the action board.
Resolvers handle the logic of executing actions during the Resolving Actions phase.

Each resolver:
- Takes the game state as input
- Provides get_valid_actions() for player choices (if any)
- Provides resolve() to execute the action and update state
- Returns a result dataclass with resolution details

Resolvers are designed to be used by the GameEngine during the resolution phase.
"""

from .starting_player import (
    StartingPlayerResolver,
    StartingPlayerResult,
)

from .buses import (
    BusesResolver,
    BusesResult,
)

from .time_clock import (
    TimeClockResolver,
    TimeClockResult,
    TimeClockAction,
)

from .passengers import (
    PassengersResolver,
    PassengersResult,
    PassengerSlotResult,
    PassengerDistribution,
)

from .buildings import (
    BuildingsResolver,
    BuildingsResult,
    BuildingSlotResult,
    BuildingPlacement,
)

from .line_expansion import (
    LineExpansionResolver,
    LineExpansionResult,
    LineExpansionSlotResult,
    RailPlacement,
)

from .vrroomm import (
    VrrooommResolver,
    VrrooommResult,
    VrrooommSlotResult,
    PassengerDelivery,
)

__all__ = [
    # Starting Player
    "StartingPlayerResolver",
    "StartingPlayerResult",
    # Buses
    "BusesResolver",
    "BusesResult",
    # Time Clock
    "TimeClockResolver",
    "TimeClockResult",
    "TimeClockAction",
    # Passengers
    "PassengersResolver",
    "PassengersResult",
    "PassengerSlotResult",
    "PassengerDistribution",
    # Buildings
    "BuildingsResolver",
    "BuildingsResult",
    "BuildingSlotResult",
    "BuildingPlacement",
    # Line Expansion
    "LineExpansionResolver",
    "LineExpansionResult",
    "LineExpansionSlotResult",
    "RailPlacement",
    # Vrroomm
    "VrrooommResolver",
    "VrrooommResult",
    "VrrooommSlotResult",
    "PassengerDelivery",
]
