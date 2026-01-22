"""Game components for the Bus game engine.

This module contains the Passenger dataclass. Buildings are represented
directly via BuildingType in BuildingSlot, and RailSegment is in board.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import NodeId


@dataclass
class Passenger:
    """Represents a passenger in the game.

    Passengers start at central parks or train stations and are transported
    via player rail networks to buildings matching the current time clock.

    Attributes:
        passenger_id: Unique identifier for this passenger.
        location: Current node ID where the passenger is located.
    """

    passenger_id: int
    location: NodeId

    def move_to(self, new_location: NodeId) -> None:
        """Move the passenger to a new node.

        Args:
            new_location: The node ID to move to.
        """
        self.location = new_location


@dataclass
class PassengerManager:
    """Manages all passengers in the game.

    Provides methods for creating, tracking, and querying passengers.
    Maintains a mapping of passenger IDs to Passenger objects.
    """

    passengers: dict[int, Passenger]
    _next_id: int = 0

    def __init__(self) -> None:
        """Initialize an empty passenger manager."""
        self.passengers = {}
        self._next_id = 0

    def create_passenger(self, location: NodeId) -> Passenger:
        """Create a new passenger at the specified location.

        Args:
            location: The node ID where the passenger starts.

        Returns:
            The newly created Passenger.
        """
        passenger = Passenger(passenger_id=self._next_id, location=location)
        self.passengers[self._next_id] = passenger
        self._next_id += 1
        return passenger

    def get_passenger(self, passenger_id: int) -> Optional[Passenger]:
        """Get a passenger by ID.

        Args:
            passenger_id: The ID of the passenger to retrieve.

        Returns:
            The Passenger if found, None otherwise.
        """
        return self.passengers.get(passenger_id)

    def get_passengers_at(self, location: NodeId) -> list[Passenger]:
        """Get all passengers at a specific node.

        Args:
            location: The node ID to query.

        Returns:
            List of passengers at that location.
        """
        return [p for p in self.passengers.values() if p.location == location]

    def move_passenger(self, passenger_id: int, new_location: NodeId) -> None:
        """Move a passenger to a new location.

        Args:
            passenger_id: The ID of the passenger to move.
            new_location: The node ID to move to.

        Raises:
            KeyError: If passenger does not exist.
        """
        passenger = self.passengers.get(passenger_id)
        if passenger is None:
            raise KeyError(f"Passenger {passenger_id} not found")
        passenger.move_to(new_location)

    def count(self) -> int:
        """Return the total number of passengers."""
        return len(self.passengers)

    def get_all_locations(self) -> dict[NodeId, list[int]]:
        """Get a mapping of node IDs to passenger IDs at each location.

        Returns:
            Dict mapping node IDs to lists of passenger IDs.
        """
        locations: dict[NodeId, list[int]] = {}
        for passenger in self.passengers.values():
            if passenger.location not in locations:
                locations[passenger.location] = []
            locations[passenger.location].append(passenger.passenger_id)
        return locations
