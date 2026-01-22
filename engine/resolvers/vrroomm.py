"""Vrroomm! action resolver for the Bus game engine.

This resolver handles the Vrroomm! action area. Players who placed
markers here get to transport passengers along their rail network
and score points by delivering them to matching buildings.

Resolution rules:
- One passenger per bus can be transported per Vrroomm! action
- Passengers can travel unlimited distance along the player's rail network
- Only deliver to buildings matching the current time clock type
- Score 1 point per delivery
- Delivered passengers remain at the destination node (not removed from game)
- During Vrroomm! resolution, delivered passengers temporarily occupy the
  building slot, preventing additional deliveries to that slot until all
  Vrroomm! actions complete
- Passengers already at a node with a matching building type are considered
  to occupy that building during this phase (auto-delivery)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.constants import ActionAreaType, BuildingType
from core.board import NodeId

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


# Slot labels in resolution order
SLOT_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@dataclass
class PassengerDelivery:
    """Represents a single passenger delivery action.

    Attributes:
        passenger_id: The passenger being transported.
        from_node: The node where the passenger started.
        to_node: The destination node.
        building_slot_index: The index of the building slot being delivered to.
    """

    passenger_id: int
    from_node: NodeId
    to_node: NodeId
    building_slot_index: int


@dataclass
class VrrooommSlotResult:
    """Result of resolving one slot in the Vrroomm! area.

    Attributes:
        player_id: The player who resolved this slot.
        slot_label: The slot label (A, B, C, etc.).
        deliveries: List of passenger deliveries made.
        points_scored: Points scored from deliveries.
    """

    player_id: int
    slot_label: str
    deliveries: list[PassengerDelivery]
    points_scored: int


@dataclass
class VrrooommResult:
    """Result of resolving all markers in the Vrroomm! area.

    Attributes:
        resolved: Whether any markers were resolved.
        slot_results: List of results for each resolved slot.
        total_points_scored: Total points scored across all slots.
        total_deliveries: Total number of deliveries made.
    """

    resolved: bool
    slot_results: list[VrrooommSlotResult] = field(default_factory=list)
    total_points_scored: int = 0
    total_deliveries: int = 0


class VrrooommResolver:
    """Resolves the Vrroomm! action area.

    This resolver:
    1. Gets all markers in the Vrroomm! area in resolution order
    2. For each marker, allows the player to transport up to buses passengers
    3. Passengers move along the player's rail network to matching buildings
    4. Each delivery scores 1 point

    The number of deliveries is limited by the player's bus count.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state
        self._current_slot_idx = 0
        self._deliveries_in_current_slot = 0
        self._slot_results: list[VrrooommSlotResult] = []
        self._current_deliveries: list[PassengerDelivery] = []
        # Track which building slots are occupied during this resolution phase
        self._occupied_slots: set[tuple[NodeId, int]] = set()
        # Auto-occupy passengers already at matching buildings
        self._mark_existing_passengers_as_occupying()

    def _mark_existing_passengers_as_occupying(self) -> None:
        """Mark building slots as occupied by passengers already at matching nodes.

        Passengers already present at a node where the matching Time Clock building
        is present are considered to occupy the building slot during this phase.
        """
        current_building_type = self.state.global_state.time_clock_position

        for node_id, node in self.state.board.nodes.items():
            # Check if node has the matching building type
            matching_slots = node.get_buildings_of_type(current_building_type)
            if not matching_slots:
                continue

            # Check if there are passengers at this node
            if not node.passenger_ids:
                continue

            # For each passenger at this node, occupy a matching building slot
            # Sort for deterministic assignment to ensure consistent slot mapping
            passenger_list = sorted(list(node.passenger_ids))
            slot_indices = [
                i for i, slot in enumerate(node.building_slots)
                if slot.building == current_building_type
            ]

            for i, passenger_id in enumerate(passenger_list):
                if i < len(slot_indices):
                    slot_idx = slot_indices[i]
                    slot = node.building_slots[slot_idx]
                    # Mark in our tracking set
                    self._occupied_slots.add((node_id, slot_idx))
                    # Also mark in the actual slot
                    if slot.occupied_by_passenger_id != passenger_id:
                        slot.occupy(passenger_id)

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Vrroomm! area in resolution order.

        Returns:
            List of occupied slots in resolution order (left-to-right).
        """
        return self.state.action_board.get_markers_to_resolve(
            ActionAreaType.VRROOMM
        )

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def get_current_slot(self) -> ActionSlot | None:
        """Get the current slot being resolved.

        Returns:
            The current slot, or None if all slots are resolved.
        """
        markers = self.get_markers_to_resolve()
        if self._current_slot_idx >= len(markers):
            return None
        return markers[self._current_slot_idx]

    def get_deliveries_remaining_for_current_slot(self) -> int:
        """Get how many more deliveries can be made for the current slot.

        Limited by the player's bus count.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            return 0

        player = self.state.get_player(current_slot.player_id)
        return player.buses - self._deliveries_in_current_slot

    def get_current_building_type(self) -> BuildingType:
        """Get the current time clock building type."""
        return self.state.global_state.time_clock_position

    def get_reachable_nodes_for_player(self, player_id: int) -> set[NodeId]:
        """Get all nodes reachable via the player's rail network.

        Returns the union of reachable nodes from all nodes in the network.

        Args:
            player_id: The player whose network to analyze.

        Returns:
            Set of all node IDs in the player's connected network.
        """
        return self.state.board.get_player_network_nodes(player_id)

    def get_available_passengers(self, player_id: int) -> list[int]:
        """Get passengers that can be transported by this player.

        A passenger is available if:
        1. They are at a node connected to the player's rail network
        2. They are not currently occupying a building slot

        Args:
            player_id: The player who would transport the passenger.

        Returns:
            List of available passenger IDs.
        """
        reachable_nodes = self.get_reachable_nodes_for_player(player_id)
        available: list[int] = []

        for passenger in self.state.passenger_manager.passengers.values():
            # Must be at a reachable node
            if passenger.location not in reachable_nodes:
                continue

            # Check if already occupying a building slot
            is_occupying = False
            for (node_id, slot_idx) in self._occupied_slots:
                if node_id == passenger.location:
                    slot = self.state.board.get_node(node_id).building_slots[slot_idx]
                    if slot.occupied_by_passenger_id == passenger.passenger_id:
                        is_occupying = True
                        break

            if not is_occupying:
                available.append(passenger.passenger_id)

        return available

    def get_available_destinations(
        self, player_id: int, passenger_id: int
    ) -> list[tuple[NodeId, int]]:
        """Get valid destinations for a passenger.

        A destination is valid if:
        1. It's reachable via the player's rail network
        2. It has a building matching the current time clock type
        3. The building slot is not already occupied

        Args:
            player_id: The player transporting the passenger.
            passenger_id: The passenger being transported.

        Returns:
            List of (node_id, slot_index) tuples for valid destinations.
        """
        passenger = self.state.passenger_manager.get_passenger(passenger_id)
        if passenger is None:
            return []

        current_building_type = self.get_current_building_type()
        reachable = self.state.board.get_reachable_nodes(
            player_id, passenger.location
        )

        destinations: list[tuple[NodeId, int]] = []

        for node_id in reachable:
            node = self.state.board.get_node(node_id)

            # Find matching building slots that aren't occupied
            for slot_idx, slot in enumerate(node.building_slots):
                if slot.building != current_building_type:
                    continue

                # Check if slot is already occupied
                if (node_id, slot_idx) in self._occupied_slots:
                    continue

                destinations.append((node_id, slot_idx))

        return destinations

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Vrroomm! resolution.

        Returns all valid (passenger, destination) combinations for the current slot.

        Returns:
            List of valid action dictionaries.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            return []

        if self.get_deliveries_remaining_for_current_slot() <= 0:
            return []

        player_id = current_slot.player_id
        # Sort for deterministic behavior and to favor lowest IDs
        available_passengers = sorted(self.get_available_passengers(player_id))

        actions = []
        seen_triples = set()  # (from_node, to_node, slot_idx)

        for passenger_id in available_passengers:
            passenger = self.state.passenger_manager.get_passenger(passenger_id)
            from_node = passenger.location
            
            destinations = self.get_available_destinations(player_id, passenger_id)
            for node_id, slot_idx in destinations:
                triple = (from_node, node_id, slot_idx)
                if triple not in seen_triples:
                    actions.append({
                        "player_id": player_id,
                        "passenger_id": passenger_id,
                        "from_node": from_node,
                        "to_node": node_id,
                        "building_slot_index": slot_idx,
                    })
                    seen_triples.add(triple)

        return actions

    def deliver_passenger(self, delivery: PassengerDelivery) -> None:
        """Execute a single passenger delivery.

        Args:
            delivery: The delivery to execute.

        Raises:
            ValueError: If the delivery is invalid.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            raise ValueError("No slot to resolve")

        if self.get_deliveries_remaining_for_current_slot() <= 0:
            raise ValueError("No deliveries remaining for current slot")

        player_id = current_slot.player_id
        passenger = self.state.passenger_manager.get_passenger(delivery.passenger_id)

        if passenger is None:
            raise ValueError(f"Passenger {delivery.passenger_id} not found")

        # Validate passenger is at the correct starting node
        if passenger.location != delivery.from_node:
            raise ValueError(
                f"Passenger {delivery.passenger_id} is at {passenger.location}, "
                f"not {delivery.from_node}"
            )

        # Validate destination is reachable
        reachable = self.state.board.get_reachable_nodes(player_id, delivery.from_node)
        if delivery.to_node not in reachable:
            raise ValueError(
                f"Node {delivery.to_node} is not reachable from {delivery.from_node} "
                f"via player {player_id}'s network"
            )

        # Validate destination has matching building
        dest_node = self.state.board.get_node(delivery.to_node)
        current_building_type = self.get_current_building_type()

        if delivery.building_slot_index >= len(dest_node.building_slots):
            raise ValueError(
                f"Invalid building slot index {delivery.building_slot_index} "
                f"for node {delivery.to_node}"
            )

        slot = dest_node.building_slots[delivery.building_slot_index]
        if slot.building != current_building_type:
            raise ValueError(
                f"Building slot has {slot.building}, expected {current_building_type}"
            )

        # Validate slot is not occupied
        if (delivery.to_node, delivery.building_slot_index) in self._occupied_slots or slot.occupied_by_passenger_id is not None:
            raise ValueError(
                f"Building slot at node {delivery.to_node} index "
                f"{delivery.building_slot_index} is already occupied"
            )

        # Execute the delivery
        # 1. Remove passenger from old node
        from_node = self.state.board.get_node(delivery.from_node)
        from_node.remove_passenger(delivery.passenger_id)

        # 2. Move passenger to new node
        self.state.passenger_manager.move_passenger(delivery.passenger_id, delivery.to_node)
        dest_node.add_passenger(delivery.passenger_id)

        # 3. Mark building slot as occupied
        slot.occupy(delivery.passenger_id)
        self._occupied_slots.add((delivery.to_node, delivery.building_slot_index))

        # 4. Score 1 point for the player
        player = self.state.get_player(player_id)
        player.add_score(1)

        # Track the delivery
        self._current_deliveries.append(delivery)
        self._deliveries_in_current_slot += 1

    def finalize_current_slot(self) -> None:
        """Finalize the current slot and move to the next."""
        current_slot = self.get_current_slot()
        if current_slot is None:
            return

        result = VrrooommSlotResult(
            player_id=current_slot.player_id,
            slot_label=current_slot.label,
            deliveries=list(self._current_deliveries),
            points_scored=len(self._current_deliveries),
        )
        self._slot_results.append(result)

        # Reset for next slot
        self._current_slot_idx += 1
        self._deliveries_in_current_slot = 0
        self._current_deliveries = []

    def skip_remaining_deliveries(self) -> None:
        """Skip remaining deliveries for the current slot and move to next.

        Call this when the player chooses not to make more deliveries
        or when no valid deliveries are available.
        """
        self.finalize_current_slot()

    def resolve_all(
        self, deliveries_per_slot: list[list[PassengerDelivery]] | None = None
    ) -> VrrooommResult:
        """Resolve all markers in the Vrroomm! area.

        If deliveries are not provided, uses a default strategy
        (make as many valid deliveries as possible).

        Args:
            deliveries_per_slot: Optional list of deliveries for each slot.

        Returns:
            VrrooommResult with all resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            return VrrooommResult(resolved=False)

        for slot_idx, slot in enumerate(markers):
            player_id = slot.player_id
            player = self.state.get_player(player_id)
            max_deliveries = player.buses

            deliveries_made = 0

            for i in range(max_deliveries):
                if deliveries_per_slot and slot_idx < len(deliveries_per_slot):
                    slot_deliveries = deliveries_per_slot[slot_idx]
                    if i < len(slot_deliveries):
                        self.deliver_passenger(slot_deliveries[i])
                        deliveries_made += 1
                        continue

                # Default: make first valid delivery
                actions = self.get_valid_actions()
                if actions:
                    action = actions[0]
                    delivery = PassengerDelivery(
                        passenger_id=action["passenger_id"],
                        from_node=action["from_node"],
                        to_node=action["to_node"],
                        building_slot_index=action["building_slot_index"],
                    )
                    self.deliver_passenger(delivery)
                    deliveries_made += 1
                else:
                    # No more valid deliveries
                    break

            # Finalize this slot
            self.finalize_current_slot()

        # Update current slot index to reflect completion after loop
        self._current_slot_idx = len(markers)

        total_points = sum(r.points_scored for r in self._slot_results)
        total_deliveries = sum(len(r.deliveries) for r in self._slot_results)

        # Cleanup: clear building slot occupancy markers
        self.clear_occupancy()

        return VrrooommResult(
            resolved=True,
            slot_results=list(self._slot_results),
            total_points_scored=total_points,
            total_deliveries=total_deliveries,
        )

    def clear_occupancy(self) -> None:
        """Clear all building slot occupancy markers after resolution.

        The occupancy was temporary during Vrroomm! resolution.
        """
        for (node_id, slot_idx) in self._occupied_slots:
            node = self.state.board.get_node(node_id)
            slot = node.building_slots[slot_idx]
            slot.vacate()
        self._occupied_slots.clear()

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        Returns True when all markers have been resolved.
        """
        markers = self.get_markers_to_resolve()
        return self._current_slot_idx >= len(markers)
