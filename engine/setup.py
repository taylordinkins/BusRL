"""Initial game setup logic for the Bus game engine.

Handles the setup phase which occurs once at the beginning of each game:
1. Place initial passengers at central parks
2. Setup building placement (SETUP_BUILDINGS phase)
3. First rail segment placement (SETUP_RAILS_FORWARD phase)
4. Second rail segment placement (SETUP_RAILS_REVERSE phase)

After setup completes, the game enters the main loop at CHOOSING_ACTIONS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.constants import (
    Phase,
    Zone,
    BuildingType,
    ZONE_PRIORITY,
    INITIAL_PASSENGERS_AT_PARKS,
    TOTAL_BUILDINGS_PER_TYPE,
)
from core.board import BoardGraph, NodeId, EdgeId, make_edge_id

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.player import Player


@dataclass
class SetupAction:
    """Represents an action taken during setup.

    Attributes:
        player_id: The player taking the action.
        action_type: Type of setup action ('building', 'rail').
        details: Additional details about the action.
    """

    player_id: int
    action_type: str
    details: dict


@dataclass
class SetupValidationResult:
    """Result of validating a setup action.

    Attributes:
        valid: Whether the action is valid.
        reason: Description of why the action is invalid (if it is).
    """

    valid: bool
    reason: Optional[str] = None


class SetupManager:
    """Manages the game setup process.

    The setup process has three sub-phases:
    1. SETUP_BUILDINGS: Players place 2 buildings each in zone A
    2. SETUP_RAILS_FORWARD: Players place 1 rail each (any edge)
    3. SETUP_RAILS_REVERSE: Players place 1 rail each (extending network)

    This class provides validation and execution of setup actions.
    """

    # Number of buildings each player places during setup
    BUILDINGS_PER_PLAYER = 2

    def __init__(self, state: GameState):
        """Initialize the setup manager.

        Args:
            state: The game state to manage setup for.
        """
        self.state = state
        # Track setup progress
        self._buildings_placed: dict[int, int] = {
            p.player_id: 0 for p in state.players
        }
        self._rails_placed_forward: dict[int, bool] = {
            p.player_id: False for p in state.players
        }
        self._rails_placed_reverse: dict[int, bool] = {
            p.player_id: False for p in state.players
        }

    # -------------------------------------------------------------------------
    # Initial Passenger Placement
    # -------------------------------------------------------------------------

    def place_initial_passengers(self) -> int:
        """Place initial passengers at central park locations.

        Each central park node gets INITIAL_PASSENGERS_AT_PARKS passengers.

        Returns:
            Number of passengers placed.
        """
        parks = self.state.board.get_central_parks()
        passengers_placed = 0

        for park in parks:
            for _ in range(INITIAL_PASSENGERS_AT_PARKS):
                passenger = self.state.passenger_manager.create_passenger(
                    location=park.node_id
                )
                park.add_passenger(passenger.passenger_id)
                passengers_placed += 1

        return passengers_placed

    # -------------------------------------------------------------------------
    # Setup Buildings Phase
    # -------------------------------------------------------------------------

    def get_valid_building_slots(self) -> list[tuple[NodeId, int]]:
        """Get all valid building slot positions for setup phase.

        During setup, buildings must be placed in Zone A (innermost zone).

        Returns:
            List of (node_id, slot_index) tuples for valid placements.
        """
        valid_slots: list[tuple[NodeId, int]] = []

        for node_id, node in self.state.board.nodes.items():
            for idx, slot in enumerate(node.building_slots):
                if slot.is_empty() and slot.zone == Zone.A:
                    valid_slots.append((node_id, idx))

        return valid_slots

    def validate_building_placement(
        self,
        player_id: int,
        node_id: NodeId,
        slot_index: int,
        building_type: BuildingType,
    ) -> SetupValidationResult:
        """Validate a building placement during setup.

        Args:
            player_id: The player placing the building.
            node_id: The node to place the building at.
            slot_index: The index of the building slot at the node.
            building_type: The type of building to place.

        Returns:
            SetupValidationResult indicating if the placement is valid.
        """
        # Check phase
        if self.state.phase != Phase.SETUP_BUILDINGS:
            return SetupValidationResult(
                valid=False,
                reason=f"Not in SETUP_BUILDINGS phase (current: {self.state.phase.value})",
            )

        # Check it's the player's turn
        if self.state.global_state.current_player_idx != player_id:
            return SetupValidationResult(
                valid=False,
                reason=f"Not player {player_id}'s turn",
            )

        # Check player hasn't placed all their buildings yet
        if self._buildings_placed.get(player_id, 0) >= self.BUILDINGS_PER_PLAYER:
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} has already placed {self.BUILDINGS_PER_PLAYER} buildings",
            )

        # Check node exists
        if node_id not in self.state.board.nodes:
            return SetupValidationResult(
                valid=False,
                reason=f"Node {node_id} does not exist",
            )

        node = self.state.board.nodes[node_id]

        # Check slot index is valid
        if slot_index < 0 or slot_index >= len(node.building_slots):
            return SetupValidationResult(
                valid=False,
                reason=f"Invalid slot index {slot_index} for node {node_id}",
            )

        slot = node.building_slots[slot_index]

        # Check slot is in Zone A
        if slot.zone != Zone.A:
            return SetupValidationResult(
                valid=False,
                reason=f"Setup buildings must be placed in Zone A, not {slot.zone.value}",
            )

        # Check slot is empty
        if not slot.is_empty():
            return SetupValidationResult(
                valid=False,
                reason=f"Slot already contains a building ({slot.building.value})",
            )

        # Check building supply limit
        if self.state.board.get_building_count(building_type) >= TOTAL_BUILDINGS_PER_TYPE:
            return SetupValidationResult(
                valid=False,
                reason=f"No supply remaining for {building_type.value} buildings",
            )

        return SetupValidationResult(valid=True)

    def place_building(
        self,
        player_id: int,
        node_id: NodeId,
        slot_index: int,
        building_type: BuildingType,
    ) -> SetupAction:
        """Place a building during setup.

        Args:
            player_id: The player placing the building.
            node_id: The node to place the building at.
            slot_index: The index of the building slot.
            building_type: The type of building to place.

        Returns:
            SetupAction describing the action taken.

        Raises:
            ValueError: If the placement is invalid.
        """
        validation = self.validate_building_placement(
            player_id, node_id, slot_index, building_type
        )
        if not validation.valid:
            raise ValueError(validation.reason)

        # Place the building
        slot = self.state.board.nodes[node_id].building_slots[slot_index]
        slot.place_building(building_type)

        # Track progress
        self._buildings_placed[player_id] = self._buildings_placed.get(player_id, 0) + 1

        return SetupAction(
            player_id=player_id,
            action_type="building",
            details={
                "node_id": node_id,
                "slot_index": slot_index,
                "building_type": building_type.value,
            },
        )

    def is_player_building_setup_complete(self, player_id: int) -> bool:
        """Check if a player has completed their building setup."""
        return self._buildings_placed.get(player_id, 0) >= self.BUILDINGS_PER_PLAYER

    def is_buildings_phase_complete(self) -> bool:
        """Check if all players have completed building placement."""
        return all(
            self.is_player_building_setup_complete(p.player_id)
            for p in self.state.players
        )

    # -------------------------------------------------------------------------
    # Setup Rails Forward Phase
    # -------------------------------------------------------------------------

    def get_valid_rail_edges_forward(self, player_id: int) -> list[EdgeId]:
        """Get all valid edges for first rail placement.

        During forward setup, players can place on ANY edge (including those
        with other players' rails - this is the exception for initial setup).

        Args:
            player_id: The player placing the rail.

        Returns:
            List of valid edge IDs.
        """
        valid_edges: list[EdgeId] = []

        for edge_id, edge in self.state.board.edges.items():
            # Can't place if player already has rail here
            if not edge.has_player_rail(player_id):
                valid_edges.append(edge_id)

        return valid_edges

    def validate_rail_placement_forward(
        self,
        player_id: int,
        edge_id: EdgeId,
    ) -> SetupValidationResult:
        """Validate a rail placement during forward setup.

        Args:
            player_id: The player placing the rail.
            edge_id: The edge to place the rail on.

        Returns:
            SetupValidationResult indicating if the placement is valid.
        """
        # Check phase
        if self.state.phase != Phase.SETUP_RAILS_FORWARD:
            return SetupValidationResult(
                valid=False,
                reason=f"Not in SETUP_RAILS_FORWARD phase (current: {self.state.phase.value})",
            )

        # Check it's the player's turn
        if self.state.global_state.current_player_idx != player_id:
            return SetupValidationResult(
                valid=False,
                reason=f"Not player {player_id}'s turn",
            )

        # Check player hasn't already placed their rail
        if self._rails_placed_forward.get(player_id, False):
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} has already placed their forward rail",
            )

        # Check player has rail segments remaining
        player = self.state.get_player(player_id)
        if not player.can_place_rail():
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} has no rail segments remaining",
            )

        # Normalize edge ID
        edge_id = make_edge_id(edge_id[0], edge_id[1])

        # Check edge exists
        if edge_id not in self.state.board.edges:
            return SetupValidationResult(
                valid=False,
                reason=f"Edge {edge_id} does not exist",
            )

        edge = self.state.board.edges[edge_id]

        # Check player doesn't already have rail here
        if edge.has_player_rail(player_id):
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} already has rail on edge {edge_id}",
            )

        return SetupValidationResult(valid=True)

    def place_rail_forward(
        self,
        player_id: int,
        edge_id: EdgeId,
    ) -> SetupAction:
        """Place a rail during forward setup.

        Args:
            player_id: The player placing the rail.
            edge_id: The edge to place the rail on.

        Returns:
            SetupAction describing the action taken.

        Raises:
            ValueError: If the placement is invalid.
        """
        edge_id = make_edge_id(edge_id[0], edge_id[1])
        validation = self.validate_rail_placement_forward(player_id, edge_id)
        if not validation.valid:
            raise ValueError(validation.reason)

        # Place the rail
        edge = self.state.board.edges[edge_id]
        edge.add_rail(player_id)

        # Deduct from player's inventory
        player = self.state.get_player(player_id)
        player.place_rail()

        # Track progress
        self._rails_placed_forward[player_id] = True

        # Initialize endpoints 
        # First rail placement creates two endpoints
        player.network_endpoints = {edge.edge_id[0], edge.edge_id[1]}

        return SetupAction(
            player_id=player_id,
            action_type="rail",
            details={
                "edge_id": edge_id,
                "phase": "forward",
            },
        )

    def is_rails_forward_phase_complete(self) -> bool:
        """Check if all players have completed forward rail placement."""
        return all(
            self._rails_placed_forward.get(p.player_id, False)
            for p in self.state.players
        )

    # -------------------------------------------------------------------------
    # Setup Rails Reverse Phase
    # -------------------------------------------------------------------------

    def get_valid_rail_edges_reverse(self, player_id: int) -> list[EdgeId]:
        """Get all valid edges for second rail placement.

        During reverse setup, the rail must extend from an endpoint of the
        player's existing network. Normal line expansion rules apply for
        parallel placement (can only share if no empty edges available).

        Args:
            player_id: The player placing the rail.

        Returns:
            List of valid edge IDs.
        """
        valid_edges: list[EdgeId] = []

        # Get player's network endpoints
        endpoints = self.state.board.get_player_network_endpoints(player_id)
        if not endpoints:
            # Player has no network - shouldn't happen in normal flow
            return []

        # Check edges at each endpoint
        for endpoint in endpoints:
            neighbors = self.state.board.get_neighbors(endpoint)
            empty_edges_at_endpoint = self.state.board.get_empty_edges_at_node(endpoint)

            for neighbor in neighbors:
                edge_id = make_edge_id(endpoint, neighbor)
                edge = self.state.board.edges.get(edge_id)
                if edge is None:
                    continue

                # Skip if player already has rail here
                if edge.has_player_rail(player_id):
                    continue

                # Check if edge is empty OR if sharing is allowed
                if edge.is_empty():
                    valid_edges.append(edge_id)
                else:
                    # Can share if no empty edges at this endpoint
                    if not empty_edges_at_endpoint:
                        valid_edges.append(edge_id)
                    # Or if endpoints touch another player's endpoint
                    elif self._endpoints_touch_other_player(endpoint, player_id):
                        valid_edges.append(edge_id)

        # Remove duplicates (an edge might be valid from multiple endpoints)
        return list(set(valid_edges))

    def _endpoints_touch_other_player(self, node_id: NodeId, player_id: int) -> bool:
        """Check if a node is an endpoint for another player's network."""
        for other_player in self.state.players:
            if other_player.player_id == player_id:
                continue
            other_endpoints = self.state.board.get_player_network_endpoints(
                other_player.player_id
            )
            if node_id in other_endpoints:
                return True
        return False

    def validate_rail_placement_reverse(
        self,
        player_id: int,
        edge_id: EdgeId,
    ) -> SetupValidationResult:
        """Validate a rail placement during reverse setup.

        Args:
            player_id: The player placing the rail.
            edge_id: The edge to place the rail on.

        Returns:
            SetupValidationResult indicating if the placement is valid.
        """
        # Check phase
        if self.state.phase != Phase.SETUP_RAILS_REVERSE:
            return SetupValidationResult(
                valid=False,
                reason=f"Not in SETUP_RAILS_REVERSE phase (current: {self.state.phase.value})",
            )

        # Check it's the player's turn
        if self.state.global_state.current_player_idx != player_id:
            return SetupValidationResult(
                valid=False,
                reason=f"Not player {player_id}'s turn",
            )

        # Check player hasn't already placed their rail
        if self._rails_placed_reverse.get(player_id, False):
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} has already placed their reverse rail",
            )

        # Check player has rail segments remaining
        player = self.state.get_player(player_id)
        if not player.can_place_rail():
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} has no rail segments remaining",
            )

        # Normalize edge ID
        edge_id = make_edge_id(edge_id[0], edge_id[1])

        # Check edge exists
        if edge_id not in self.state.board.edges:
            return SetupValidationResult(
                valid=False,
                reason=f"Edge {edge_id} does not exist",
            )

        edge = self.state.board.edges[edge_id]

        # Check player doesn't already have rail here
        if edge.has_player_rail(player_id):
            return SetupValidationResult(
                valid=False,
                reason=f"Player {player_id} already has rail on edge {edge_id}",
            )

        # Check edge is connected to an endpoint
        endpoints = self.state.board.get_player_network_endpoints(player_id)
        node_a, node_b = edge_id
        if node_a not in endpoints and node_b not in endpoints:
            return SetupValidationResult(
                valid=False,
                reason=f"Edge {edge_id} is not connected to player {player_id}'s network endpoints",
            )

        # Check sharing rules
        if not edge.is_empty():
            # Find which endpoint this edge connects to
            connected_endpoint = node_a if node_a in endpoints else node_b
            empty_edges = self.state.board.get_empty_edges_at_node(connected_endpoint)

            if empty_edges and not self._endpoints_touch_other_player(
                connected_endpoint, player_id
            ):
                return SetupValidationResult(
                    valid=False,
                    reason=f"Cannot share edge - empty edges available at endpoint {connected_endpoint}",
                )

        return SetupValidationResult(valid=True)

    def place_rail_reverse(
        self,
        player_id: int,
        edge_id: EdgeId,
    ) -> SetupAction:
        """Place a rail during reverse setup.

        Args:
            player_id: The player placing the rail.
            edge_id: The edge to place the rail on.

        Returns:
            SetupAction describing the action taken.

        Raises:
            ValueError: If the placement is invalid.
        """
        edge_id = make_edge_id(edge_id[0], edge_id[1])
        validation = self.validate_rail_placement_reverse(player_id, edge_id)
        if not validation.valid:
            raise ValueError(validation.reason)

        # Place the rail
        edge = self.state.board.edges[edge_id]
        edge.add_rail(player_id)

        # Deduct from player's inventory
        player = self.state.get_player(player_id)
        player.place_rail()

        # Track progress
        self._rails_placed_reverse[player_id] = True

        # Update endpoints
        # Reverse phase extends from existing endpoints
        # The new rail connects an existing endpoint to a new node (or another endpoint)
        endpoints = self.state.board.get_player_network_endpoints(player_id)
        node_a, node_b = edge_id
        
        # Identify which node was the existing endpoint
        # Note: endpoints set from board is still using the old logic which is stateless
        # We need to rely on the player's stored endpoints if they exist, or initialize them
        if not player.network_endpoints:
             # Should be initialized from forward phase, but if not (legacy/test), use board calculation
             player.network_endpoints = endpoints

        # Determine start and end of this new segment
        # One of node_a/node_b must be in current endpoints
        start_node = node_a if node_a in player.network_endpoints else node_b
        end_node = node_b if start_node == node_a else node_a

        # Update endpoints
        player.network_endpoints.remove(start_node)
        player.network_endpoints.add(end_node)
        
        return SetupAction(
            player_id=player_id,
            action_type="rail",
            details={
                "edge_id": edge_id,
                "phase": "reverse",
            },
        )

    def is_rails_reverse_phase_complete(self) -> bool:
        """Check if all players have completed reverse rail placement."""
        return all(
            self._rails_placed_reverse.get(p.player_id, False)
            for p in self.state.players
        )

    # -------------------------------------------------------------------------
    # Turn Order Helpers
    # -------------------------------------------------------------------------

    def get_buildings_player_order(self) -> list[int]:
        """Get player order for building placement (clockwise from first player)."""
        num_players = len(self.state.players)
        start = self.state.global_state.starting_player_idx
        return [(start + i) % num_players for i in range(num_players)]

    def get_rails_forward_player_order(self) -> list[int]:
        """Get player order for forward rail placement (same as building order)."""
        return self.get_buildings_player_order()

    def get_rails_reverse_player_order(self) -> list[int]:
        """Get player order for reverse rail placement (reverse of forward)."""
        return list(reversed(self.get_rails_forward_player_order()))

    def get_next_player_buildings(self) -> Optional[int]:
        """Get the next player who needs to place buildings, or None if done."""
        for player_id in self.get_buildings_player_order():
            if not self.is_player_building_setup_complete(player_id):
                return player_id
        return None

    def get_next_player_rails_forward(self) -> Optional[int]:
        """Get the next player who needs to place forward rail, or None if done."""
        for player_id in self.get_rails_forward_player_order():
            if not self._rails_placed_forward.get(player_id, False):
                return player_id
        return None

    def get_next_player_rails_reverse(self) -> Optional[int]:
        """Get the next player who needs to place reverse rail, or None if done."""
        for player_id in self.get_rails_reverse_player_order():
            if not self._rails_placed_reverse.get(player_id, False):
                return player_id
        return None

    # -------------------------------------------------------------------------
    # Setup Completion
    # -------------------------------------------------------------------------

    def is_setup_complete(self) -> bool:
        """Check if all setup phases are complete."""
        return (
            self.is_buildings_phase_complete()
            and self.is_rails_forward_phase_complete()
            and self.is_rails_reverse_phase_complete()
        )

    def get_setup_summary(self) -> dict:
        """Get a summary of setup progress.

        Returns:
            Dictionary with setup progress details.
        """
        return {
            "buildings_placed": dict(self._buildings_placed),
            "rails_forward_placed": dict(self._rails_placed_forward),
            "rails_reverse_placed": dict(self._rails_placed_reverse),
            "buildings_phase_complete": self.is_buildings_phase_complete(),
            "rails_forward_phase_complete": self.is_rails_forward_phase_complete(),
            "rails_reverse_phase_complete": self.is_rails_reverse_phase_complete(),
            "setup_complete": self.is_setup_complete(),
        }


def initialize_game(state: GameState) -> SetupManager:
    """Initialize a new game by placing initial passengers.

    This is called once at the very start of the game, before any
    player actions. It places the initial passengers at central parks.

    Args:
        state: The game state to initialize.

    Returns:
        A SetupManager ready to manage the setup phases.
    """
    manager = SetupManager(state)
    manager.place_initial_passengers()
    return manager
