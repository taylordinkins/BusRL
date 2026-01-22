"""Passengers action resolver for the Bus game engine.

This resolver handles the Passengers action area. Players who placed
markers here get to spawn new passengers at train stations.

Resolution rules:
- Number of passengers = M#oB - slot_index (A=M#oB, B=M#oB-1, etc.)
- Player chooses how to distribute spawned passengers between train stations
- If only one train station exists, all passengers go there
- Minimum 1 passenger per resolution (if M#oB - slot_index < 1, player gets 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.constants import ActionAreaType, TOTAL_PASSENGERS
from core.board import NodeId

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


# Slot labels in resolution order (A first for this area)
SLOT_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@dataclass
class PassengerDistribution:
    """Represents how passengers should be distributed to train stations.

    Attributes:
        distribution: Mapping from train station node ID to number of passengers.
    """

    distribution: dict[NodeId, int] = field(default_factory=dict)

    def total(self) -> int:
        """Get total number of passengers in this distribution."""
        return sum(self.distribution.values())


@dataclass
class PassengerSlotResult:
    """Result of resolving one slot in the Passengers area.

    Attributes:
        player_id: The player who resolved this slot.
        slot_label: The slot label (A, B, C, etc.).
        passengers_spawned: Number of passengers spawned.
        distribution: How passengers were distributed to train stations.
    """

    player_id: int
    slot_label: str
    passengers_spawned: int
    distribution: dict[NodeId, int]


@dataclass
class PassengersResult:
    """Result of resolving all markers in the Passengers area.

    Attributes:
        resolved: Whether any markers were resolved.
        slot_results: List of results for each resolved slot.
        total_passengers_spawned: Total passengers spawned across all slots.
    """

    resolved: bool
    slot_results: list[PassengerSlotResult] = field(default_factory=list)
    total_passengers_spawned: int = 0


class PassengersResolver:
    """Resolves the Passengers action area.

    This resolver:
    1. Gets all markers in the Passengers area in resolution order
    2. For each marker, calculates passengers to spawn (M#oB - slot_index)
    3. Player chooses distribution across train stations
    4. Creates passengers and places them at the chosen stations

    The number of passengers scales with M#oB and decreases for later slots.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state
        self._current_slot_idx = 0
        self._slot_results: list[PassengerSlotResult] = []

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Passengers area in resolution order.

        Returns:
            List of occupied slots in resolution order (left-to-right).
        """
        return self.state.action_board.get_markers_to_resolve(
            ActionAreaType.PASSENGERS
        )

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def get_remaining_markers(self) -> list[ActionSlot]:
        """Get markers that haven't been resolved yet."""
        markers = self.get_markers_to_resolve()
        return markers[self._current_slot_idx:]

    def get_current_slot(self) -> ActionSlot | None:
        """Get the current slot being resolved.

        Automatically skips slots that provide 0 passengers (wasted markers)
        or if the passenger limit has been reached.

        Returns:
            The current slot, or None if all slots are resolved or limit reached.
        """
        # Check if we already hit the global passenger limit
        if self.state.passenger_manager.count() >= TOTAL_PASSENGERS:
            return None

        markers = self.get_markers_to_resolve()
        
        # Advance past any empty slots
        while self._current_slot_idx < len(markers):
            slot = markers[self._current_slot_idx]
            if self.get_passengers_for_slot(slot) > 0:
                return slot
            self._current_slot_idx += 1
            
        return None

    def get_max_buses(self) -> int:
        """Get the current Maximum Number of Buses (M#oB).

        M#oB is the highest number of buses owned by any player.
        """
        return max(p.buses for p in self.state.players)

    def get_passengers_for_slot(self, slot: ActionSlot) -> int:
        """Calculate the number of passengers for a given slot.

        Formula: M#oB - slot_index (A=0, B=1, etc.)
        Can be 0 if slot_index >= M#oB (wasted marker).
        Also limited by the global passenger supply.

        Args:
            slot: The action slot being resolved.

        Returns:
            Number of passengers to spawn (0 or more).
        """
        slot_index = SLOT_TO_INDEX.get(slot.label, 0)
        passengers = self.get_max_buses() - slot_index
        requested = max(0, passengers)

        # Apply global supply limit
        current_count = self.state.passenger_manager.count()
        remaining_capacity = max(0, TOTAL_PASSENGERS - current_count)
        
        return min(requested, remaining_capacity)

    def get_train_stations(self) -> list[NodeId]:
        """Get all train station node IDs.

        Returns:
            List of node IDs that are train stations.
        """
        return [node.node_id for node in self.state.board.get_train_stations()]

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Passengers resolution.

        For the current slot, returns all valid distributions of passengers
        across the train stations.

        Returns:
            List of valid action dictionaries. Each contains:
            - player_id: The player who must make the decision
            - distribution: Dict mapping train station node IDs to passenger counts
            - total_passengers: Total passengers being placed
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            return []

        player_id = current_slot.player_id
        passengers_to_place = self.get_passengers_for_slot(current_slot)
        train_stations = self.get_train_stations()

        if not train_stations:
            # Edge case: no train stations (shouldn't happen in valid board)
            return []

        if len(train_stations) == 1:
            # Only one station - no choice, all go there
            return [{
                "player_id": player_id,
                "distribution": {train_stations[0]: passengers_to_place},
                "total_passengers": passengers_to_place,
            }]

        # Generate all valid distributions for 2 train stations
        # (the standard Bus board has exactly 2 train stations)
        actions = []
        station_a, station_b = train_stations[0], train_stations[1]

        for count_a in range(passengers_to_place + 1):
            count_b = passengers_to_place - count_a
            actions.append({
                "player_id": player_id,
                "distribution": {station_a: count_a, station_b: count_b},
                "total_passengers": passengers_to_place,
            })

        return actions

    def resolve_slot(self, distribution: PassengerDistribution) -> PassengerSlotResult:
        """Resolve the current slot with the given distribution.

        Args:
            distribution: How to distribute passengers across train stations.

        Returns:
            PassengerSlotResult with details of this slot's resolution.

        Raises:
            ValueError: If no slot to resolve or distribution is invalid.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            raise ValueError("No slot to resolve")

        player_id = current_slot.player_id
        # We use the adjusted count from get_passengers_for_slot which already handles limits
        expected_passengers = self.get_passengers_for_slot(current_slot)

        # Validate distribution
        if distribution.total() != expected_passengers:
            raise ValueError(
                f"Distribution total ({distribution.total()}) doesn't match "
                f"expected passengers ({expected_passengers})"
            )

        train_stations = set(self.get_train_stations())
        for station_id in distribution.distribution.keys():
            if station_id not in train_stations:
                raise ValueError(f"Invalid train station ID: {station_id}")

        # Spawn passengers at the specified locations
        for station_id, count in distribution.distribution.items():
            for _ in range(count):
                passenger = self.state.passenger_manager.create_passenger(station_id)
                # Also update the node's passenger_ids
                self.state.board.get_node(station_id).add_passenger(passenger.passenger_id)

        result = PassengerSlotResult(
            player_id=player_id,
            slot_label=current_slot.label,
            passengers_spawned=expected_passengers,
            distribution=dict(distribution.distribution),
        )

        self._slot_results.append(result)
        self._current_slot_idx += 1

        return result

    def resolve_all(
        self, distributions: list[PassengerDistribution] | None = None
    ) -> PassengersResult:
        """Resolve all markers in the Passengers area.

        If distributions are not provided, uses default distribution
        (split evenly between stations, with remainder to first station).

        Args:
            distributions: Optional list of distributions for each slot.

        Returns:
            PassengersResult with all resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            return PassengersResult(resolved=False)

        train_stations = self.get_train_stations()

        for i, slot in enumerate(markers):
            # Sync internal index with loop
            self._current_slot_idx = i
            
            if distributions and i < len(distributions):
                dist = distributions[i]
            else:
                # Default: split evenly (or all to first station if only one)
                passengers = self.get_passengers_for_slot(slot)
                
                if passengers == 0:
                    dist = PassengerDistribution() # Empty distribution
                elif len(train_stations) == 1:
                    dist = PassengerDistribution({train_stations[0]: passengers})
                else:
                    # Split as evenly as possible
                    half = passengers // 2
                    remainder = passengers % 2
                    dist = PassengerDistribution({
                        train_stations[0]: half + remainder,
                        train_stations[1]: half,
                    })

            # Check passengers again just in case distribution was passed manually 
            # but for a 0-passenger slot (user error, or just alignment)
            # We must ensure we don't call resolve_slot if current slot yields 0.
            if self.get_passengers_for_slot(slot) == 0:
                 # Still record a result but skip spawning
                 result = PassengerSlotResult(
                     player_id=slot.player_id,
                     slot_label=slot.label,
                     passengers_spawned=0,
                     distribution={}
                 )
                 self._slot_results.append(result)
                 self._current_slot_idx = i + 1
                 continue
                 
            self.resolve_slot(dist)

        # Update current slot index to reflect completion after loop
        self._current_slot_idx = len(markers)

        total = sum(r.passengers_spawned for r in self._slot_results)

        return PassengersResult(
            resolved=True,
            slot_results=list(self._slot_results),
            total_passengers_spawned=total,
        )

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        Returns True when all markers have been resolved.
        """
        return self._current_slot_idx >= len(self.get_markers_to_resolve())
