"""Line Expansion action resolver for the Bus game engine.

This resolver handles the Line Expansion action area. Players who placed
markers here get to extend their rail network.

Resolution rules:
- Number of segments = M#oB - slot_index (A=M#oB, B=M#oB-1, etc.)
- Must extend from an endpoint of the player's existing network
- Rail sharing rules:
  - Can only place on an edge with another player's rail if:
    1. No empty edges available at any of the player's endpoints, OR
    2. One of the player's endpoints touches another player's endpoint
  - When sharing is allowed, should generally prefer empty edges
- Segments can alternate between endpoints within one action
- Loop creation requires logical endpoint tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.constants import ActionAreaType
from core.board import NodeId, EdgeId, make_edge_id

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


# Slot labels in resolution order
SLOT_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@dataclass
class RailPlacement:
    """Represents a single rail segment placement.

    Attributes:
        edge_id: The edge where the rail is placed.
        from_endpoint: The endpoint node from which this extends.
    """

    edge_id: EdgeId
    from_endpoint: NodeId


@dataclass
class LineExpansionSlotResult:
    """Result of resolving one slot in the Line Expansion area.

    Attributes:
        player_id: The player who resolved this slot.
        slot_label: The slot label (A, B, C, etc.).
        segments_placed: Number of rail segments placed.
        placements: List of individual placements made.
    """

    player_id: int
    slot_label: str
    segments_placed: int
    placements: list[RailPlacement]


@dataclass
class LineExpansionResult:
    """Result of resolving all markers in the Line Expansion area.

    Attributes:
        resolved: Whether any markers were resolved.
        slot_results: List of results for each resolved slot.
        total_segments_placed: Total segments placed across all slots.
    """

    resolved: bool
    slot_results: list[LineExpansionSlotResult] = field(default_factory=list)
    total_segments_placed: int = 0


class LineExpansionResolver:
    """Resolves the Line Expansion action area.

    This resolver:
    1. Gets all markers in the Line Expansion area in resolution order
    2. For each marker, calculates segments to place (M#oB - slot_index)
    3. Player places segments one at a time, extending from endpoints
    4. Enforces rail sharing rules

    The number of segments scales with M#oB and decreases for later slots.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state
        self._current_slot_idx = 0
        self._segments_placed_in_current_slot = 0
        self._slot_results: list[LineExpansionSlotResult] = []
        self._current_placements: list[RailPlacement] = []

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Line Expansion area in resolution order.

        Returns:
            List of occupied slots in resolution order (left-to-right).
        """
        return self.state.action_board.get_markers_to_resolve(
            ActionAreaType.LINE_EXPANSION
        )

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def get_current_slot(self) -> ActionSlot | None:
        """Get the current slot being resolved.

        Automatically skips:
        1. Slots that provide 0 segments (wasted markers)
        2. Slots belonging to players who have run out of rail segments.

        Returns:
            The current slot, or None if all slots are resolved.
        """
        markers = self.get_markers_to_resolve()
        
        # Advance past any empty slots or slots for players without rails
        while self._current_slot_idx < len(markers):
            slot = markers[self._current_slot_idx]
            player = self.state.get_player(slot.player_id)
            
            if self.get_segments_for_slot(slot) > 0 and player.can_place_rail():
                return slot
            
            # This slot cannot be resolved (either 0 segments or no rails left)
            # Advance to the next marker
            self._current_slot_idx += 1
            
        return None

    def get_max_buses(self) -> int:
        """Get the current Maximum Number of Buses (M#oB)."""
        return max(p.buses for p in self.state.players)

    def get_segments_for_slot(self, slot: ActionSlot) -> int:
        """Calculate the number of segments for a given slot.

        Formula: M#oB - slot_index (A=0, B=1, etc.)
        Can be 0 if slot_index >= M#oB (wasted marker).
        Also limited by the player's remaining rail segments.

        Args:
            slot: The action slot being resolved.

        Returns:
            Number of segments to place (0 or more).
        """
        slot_index = SLOT_TO_INDEX.get(slot.label, 0)
        requested = max(0, self.get_max_buses() - slot_index)

        # Apply player's rail limit
        player = self.state.get_player(slot.player_id)
        return min(requested, player.rail_segments_remaining)

    def get_segments_remaining_for_current_slot(self) -> int:
        """Get how many segments still need to be placed for the current slot."""
        current_slot = self.get_current_slot()
        if current_slot is None:
            return 0
        total = self.get_segments_for_slot(current_slot)
        return total - self._segments_placed_in_current_slot

    def get_player_endpoints(self, player_id: int) -> set[NodeId]:
        """Get the current endpoints of a player's rail network.

        An endpoint is a node where the player has exactly one connected edge.
        If the network forms a loop (no endpoints), all nodes are endpoints.

        Args:
            player_id: The player whose endpoints to find.

        Returns:
            Set of endpoint node IDs.
        """
        return self.state.get_player(player_id).network_endpoints

    def _get_other_players_endpoints(self, player_id: int) -> set[NodeId]:
        """Get all endpoints of other players' rail networks.

        Args:
            player_id: The player to exclude.

        Returns:
            Set of endpoint node IDs belonging to other players.
        """
        other_endpoints: set[NodeId] = set()
        for player in self.state.players:
            if player.player_id != player_id:
                other_endpoints.update(self.get_player_endpoints(player.player_id))
        return other_endpoints

    def _has_empty_edge_at_endpoints(self, player_id: int) -> bool:
        """Check if the player has any empty edge at any of their endpoints.

        Args:
            player_id: The player to check.

        Returns:
            True if at least one endpoint has an empty adjacent edge.
        """
        endpoints = self.get_player_endpoints(player_id)
        for endpoint in endpoints:
            empty_edges = self.state.board.get_empty_edges_at_node(endpoint)
            if empty_edges:
                return True
        return False

    def _endpoints_touch_other_player(self, player_id: int) -> bool:
        """Check if any of the player's endpoints touch another player's endpoint.

        Args:
            player_id: The player to check.

        Returns:
            True if any endpoint is shared with another player.
        """
        player_endpoints = self.get_player_endpoints(player_id)
        other_endpoints = self._get_other_players_endpoints(player_id)
        return bool(player_endpoints & other_endpoints)

    def can_share_rails(self, player_id: int) -> bool:
        """Check if the player is allowed to place on shared edges.

        Rail sharing is allowed if:
        1. No empty edges available at any endpoint, OR
        2. One of the player's endpoints touches another player's endpoint

        Args:
            player_id: The player to check.

        Returns:
            True if rail sharing is allowed.
        """
        # Rule 1: No empty edges at any endpoint
        if not self._has_empty_edge_at_endpoints(player_id):
            return True

        # Rule 2: Endpoints touch another player's endpoint
        if self._endpoints_touch_other_player(player_id):
            return True

        return False

    def get_valid_edges_from_endpoint(
        self, player_id: int, endpoint: NodeId
    ) -> list[EdgeId]:
        """Get valid edges for expansion from a specific endpoint.

        Args:
            player_id: The player expanding their network.
            endpoint: The endpoint node to expand from.

        Returns:
            List of valid edge IDs for placement.
        """
        valid_edges: list[EdgeId] = []
        can_share = self.can_share_rails(player_id)

        for neighbor in self.state.board.get_neighbors(endpoint):
            edge_id = make_edge_id(endpoint, neighbor)
            edge = self.state.board.edges.get(edge_id)

            if edge is None:
                continue

            # Already has this player's rail
            if edge.has_player_rail(player_id):
                continue

            # Check if sharing is required
            if not edge.is_empty():
                # Edge has another player's rail
                if can_share:
                    valid_edges.append(edge_id)
                # else: can't place here
            else:
                # Empty edge - always valid
                valid_edges.append(edge_id)

        return valid_edges

    def get_valid_placements(self, player_id: int) -> list[RailPlacement]:
        """Get all valid rail placements for the player.

        Args:
            player_id: The player placing rails.

        Returns:
            List of valid RailPlacement options.
        """
        endpoints = self.get_player_endpoints(player_id)
        placements: list[RailPlacement] = []

        for endpoint in endpoints:
            valid_edges = self.get_valid_edges_from_endpoint(player_id, endpoint)
            for edge_id in valid_edges:
                placements.append(RailPlacement(
                    edge_id=edge_id,
                    from_endpoint=endpoint,
                ))

        return placements

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Line Expansion resolution.

        Returns:
            List of valid action dictionaries.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            return []

        if self.get_segments_remaining_for_current_slot() <= 0:
            return []

        player_id = current_slot.player_id
        player = self.state.get_player(player_id)

        # Check if player has rail segments remaining
        if not player.can_place_rail():
            return []

        placements = self.get_valid_placements(player_id)

        return [
            {
                "player_id": player_id,
                "edge_id": p.edge_id,
                "from_endpoint": p.from_endpoint,
            }
            for p in placements
        ]

    def place_rail(self, placement: RailPlacement) -> None:
        """Place a single rail segment on the board.

        Args:
            placement: The rail placement to execute.

        Raises:
            ValueError: If the placement is invalid.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            raise ValueError("No slot to resolve")

        if self.get_segments_remaining_for_current_slot() <= 0:
            raise ValueError("No segments remaining for current slot")

        player_id = current_slot.player_id
        player = self.state.get_player(player_id)

        if not player.can_place_rail():
            raise ValueError(f"Player {player_id} has no rail segments remaining")

        # Validate the placement
        endpoints = self.get_player_endpoints(player_id)
        if placement.from_endpoint not in endpoints:
            raise ValueError(
                f"Node {placement.from_endpoint} is not an endpoint of player {player_id}'s network"
            )

        valid_edges = self.get_valid_edges_from_endpoint(player_id, placement.from_endpoint)
        if placement.edge_id not in valid_edges:
            raise ValueError(
                f"Edge {placement.edge_id} is not a valid expansion from endpoint {placement.from_endpoint}"
            )

        # Place the rail
        edge = self.state.board.edges[placement.edge_id]
        edge.add_rail(player_id)
        player.place_rail()

        # Track the placement
        self._current_placements.append(placement)
        self._segments_placed_in_current_slot += 1

        # Update player endpoints
        # We extended from placement.from_endpoint to placement.to_node (derived)
        node_a, node_b = placement.edge_id
        to_node = node_a if node_a != placement.from_endpoint else node_b
        
        # Remove the endpoint we extended from
        if placement.from_endpoint in player.network_endpoints:
            player.network_endpoints.remove(placement.from_endpoint)
        
        # Add the new endpoint
        # If we connected to another existing endpoint (closing a loop), 
        # that node becomes the "join point" and is logically an endpoint for further expansion?
        # The user rule: "the point where the loop occurs is the only valid endpoint"
        # If to_node was ALREADY in network_endpoints (e.g. we connected two tips),
        # then we just merged two branches. Both tips are consumed?
        # Wait, if I have U-V and X-Y-Z. Endpoints {U, V, X, Z}.
        # If I connect V-X. New Endpoints {U, Z}. V and X are internal now.
        # But here we are always extending from an endpoint.
        # If to_node is in network_endpoints => We connected two endpoints -> Loop/Merge.
        # If to_node is NOT in network_endpoints => Simple extension.
        
        # However, the user specific request: "when a closed loop occurs, the point where the loop occurs is the only valid endpoint"
        # "Point where loop occurs" usually means the node we just connected TO.
        # So if we connect endpoint A to endpoint B. B becomes the single active endpoint?
        # Or does the user mean if we connect A to some internal node B?
        # But we can't connect to internal B unless we share rail?
        # Assuming we connect to an endpoint B (closing a big cycle):
        # The user wants "that point" to be valid.
        
        # Let's assume standard behavior:
        # If simple extension: Remove From, Add To.
        # If closing loop (To is already in Endpoints): Remove From, Remove To? 
        # BUT User says: "point where loop occurs is the only valid endpoint".
        # This implies we should KEEP 'To' as an endpoint if it was one.
        # Actually, if we close a loop, the graph has NO topological endpoints.
        # The user wants us to ARTIFICIALLY mark the closure point as an endpoint.
        
        if to_node in player.network_endpoints:
             # We connected two endpoints. 
             # Is this a loop closure? yes.
             # User wants "the point where the loop occurs" (to_node?) to be the endpoint.
             # So we do NOT remove to_node. We just remove from_node.
             pass
        else:
             # Check if we connected to an internal node (sharing rail with self? allowed?)
             # Usually not allowed to overlap own rail.
             # So to_node is a new node or an existing node not in endpoints.
             # If it's an existing node not in endpoints, it's internal. 
             # Connection to internal node implies cycle?
             # But we can't connect to internal node unless we overlap, which is banned.
             # So to_node must be a new node or another endpoint.
             player.network_endpoints.add(to_node)

        # Check if we've placed all segments for this slot
        if self._segments_placed_in_current_slot >= self.get_segments_for_slot(current_slot):
            self._finalize_current_slot()

    def _finalize_current_slot(self) -> None:
        """Finalize the current slot and move to the next."""
        current_slot = self.get_current_slot()
        if current_slot is None:
            return

        result = LineExpansionSlotResult(
            player_id=current_slot.player_id,
            slot_label=current_slot.label,
            segments_placed=len(self._current_placements),
            placements=list(self._current_placements),
        )
        self._slot_results.append(result)

        # Reset for next slot
        self._current_slot_idx += 1
        self._segments_placed_in_current_slot = 0
        self._current_placements = []

    def resolve_all(
        self, placements_per_slot: list[list[RailPlacement]] | None = None
    ) -> LineExpansionResult:
        """Resolve all markers in the Line Expansion area.

        If placements are not provided, uses default placement
        (first valid edge from first endpoint for each segment).

        Args:
            placements_per_slot: Optional list of placements for each slot.

        Returns:
            LineExpansionResult with all resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            return LineExpansionResult(resolved=False)

        # Reset state to ensure clean resolution
        self._current_slot_idx = 0
        self._segments_placed_in_current_slot = 0
        self._slot_results = []
        self._current_placements = []

        for slot_idx, slot in enumerate(markers):
            # Sync internal slot index with loop index
            self._current_slot_idx = slot_idx
            self._segments_placed_in_current_slot = 0
            self._current_placements = []

            player_id = slot.player_id
            player = self.state.get_player(player_id)
            segments_to_place = self.get_segments_for_slot(slot)

            # Skip if no segments to place or player can't place rails
            if segments_to_place <= 0 or not player.can_place_rail():
                self._finalize_current_slot()
                continue

            for i in range(segments_to_place):
                # Check if player has rails remaining
                if not player.can_place_rail():
                    break

                if placements_per_slot and slot_idx < len(placements_per_slot):
                    slot_placements = placements_per_slot[slot_idx]
                    if i < len(slot_placements):
                        self._place_rail_internal(slot, slot_placements[i])
                        continue

                # Default: place on first valid edge
                valid_placements = self.get_valid_placements(player_id)
                if valid_placements:
                    # Prefer empty edges over shared edges
                    empty_edge_placements = [
                        p for p in valid_placements
                        if self.state.board.edges[p.edge_id].is_empty()
                    ]
                    if empty_edge_placements:
                        self._place_rail_internal(slot, empty_edge_placements[0])
                    else:
                        self._place_rail_internal(slot, valid_placements[0])
                else:
                    # No valid placements available
                    break

            # Always finalize the slot
            self._finalize_current_slot()

        total = sum(r.segments_placed for r in self._slot_results)

        return LineExpansionResult(
            resolved=True,
            slot_results=list(self._slot_results),
            total_segments_placed=total,
        )

    def _place_rail_internal(self, slot: "ActionSlot", placement: RailPlacement) -> None:
        """Internal method to place rail without slot validation.

        Used by resolve_all() which manages slot iteration itself.

        Args:
            slot: The slot being resolved.
            placement: The rail placement to execute.
        """
        player_id = slot.player_id
        player = self.state.get_player(player_id)

        # Place the rail
        edge = self.state.board.edges[placement.edge_id]
        edge.add_rail(player_id)
        player.place_rail()

        # Track the placement
        self._current_placements.append(placement)
        self._segments_placed_in_current_slot += 1

        # Update player endpoints (same logic as place_rail method)
        node_a, node_b = placement.edge_id
        to_node = node_a if node_a != placement.from_endpoint else node_b

        # Remove the endpoint we extended from
        if placement.from_endpoint in player.network_endpoints:
            player.network_endpoints.remove(placement.from_endpoint)

        # Add the new endpoint unless it's already an endpoint (loop closure)
        # If to_node is already in endpoints, we're closing a loop - keep it as endpoint
        # If to_node is not in endpoints, it's a new extension point
        if to_node not in player.network_endpoints:
            player.network_endpoints.add(to_node)

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        Returns True when all markers have been resolved.
        """
        markers = self.get_markers_to_resolve()
        return self._current_slot_idx >= len(markers)
