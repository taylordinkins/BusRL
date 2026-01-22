"""Board graph model for the Bus game engine.

The board is represented as a static attributed graph:
- Nodes represent intersections with building slots and passenger locations
- Edges represent streets where rail segments may be built
- Topology is immutable; only occupancy (buildings, passengers, rails) changes
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from .constants import BuildingType, Zone


# Type aliases for clarity
NodeId = int
EdgeId = tuple[int, int]  # Canonical form: (min_id, max_id)


def make_edge_id(node_a: int, node_b: int) -> EdgeId:
    """Create a canonical edge ID from two node IDs.

    Edge IDs are always stored with the smaller node ID first
    to ensure consistent lookups regardless of direction.
    """
    return (min(node_a, node_b), max(node_a, node_b))


@dataclass
class BuildingSlot:
    """A single building slot at a node.

    Each slot belongs to a specific zone (A, B, C, or D).
    A node may have multiple slots, potentially from different zones.
    """

    zone: Zone
    building: Optional[BuildingType] = None
    occupied_by_passenger_id: Optional[int] = None

    def is_empty(self) -> bool:
        """Check if this slot has no building placed."""
        return self.building is None

    def is_occupied_by_passenger(self) -> bool:
        """Check if this slot is temporarily occupied by a passenger."""
        return self.occupied_by_passenger_id is not None

    def occupy(self, passenger_id: int) -> None:
        """Mark this slot as occupied by a passenger (Vrroomm! phase).

        Args:
            passenger_id: The ID of the passenger occupying the slot.

        Raises:
            ValueError: If already occupied by a passenger.
        """
        if self.occupied_by_passenger_id is not None:
            raise ValueError(f"Slot already occupied by passenger {self.occupied_by_passenger_id}")
        self.occupied_by_passenger_id = passenger_id

    def vacate(self) -> None:
        """Remove passenger occupancy."""
        self.occupied_by_passenger_id = None

    def place_building(self, building_type: BuildingType) -> None:
        """Place a building in this slot.

        Raises:
            ValueError: If slot already has a building.
        """
        if self.building is not None:
            raise ValueError(f"Slot already contains a {self.building}")
        self.building = building_type

    def clone(self) -> BuildingSlot:
        """Create a deep copy of this building slot."""
        return BuildingSlot(
            zone=self.zone,
            building=self.building,
            occupied_by_passenger_id=self.occupied_by_passenger_id
        )


@dataclass
class NodeState:
    """State of a single node (intersection) on the board.

    Attributes:
        node_id: Unique identifier for this node.
        building_slots: List of building slots at this node.
        passenger_ids: Set of passenger IDs currently at this node.
        is_train_station: True if this node is a train station (passenger spawn point).
        is_central_park: True if this node is a central park (initial passenger location).
        position: (x, y) coordinates for visualization.
    """

    node_id: NodeId
    building_slots: list[BuildingSlot] = field(default_factory=list)
    passenger_ids: set[int] = field(default_factory=set)
    is_train_station: bool = False
    is_central_park: bool = False
    position: tuple[float, float] = (0.0, 0.0)

    def get_empty_slots(self) -> list[BuildingSlot]:
        """Return all building slots without a building."""
        return [slot for slot in self.building_slots if slot.is_empty()]

    def get_empty_slots_by_zone(self, zone: Zone) -> list[BuildingSlot]:
        """Return empty building slots of a specific zone."""
        return [
            slot for slot in self.building_slots
            if slot.is_empty() and slot.zone == zone
        ]

    def get_buildings_of_type(self, building_type: BuildingType) -> list[BuildingSlot]:
        """Return all slots containing a specific building type."""
        return [
            slot for slot in self.building_slots
            if slot.building == building_type
        ]

    def has_building_type(self, building_type: BuildingType) -> bool:
        """Check if this node has at least one building of the given type."""
        return any(slot.building == building_type for slot in self.building_slots)

    def add_passenger(self, passenger_id: int) -> None:
        """Add a passenger to this node."""
        self.passenger_ids.add(passenger_id)

    def remove_passenger(self, passenger_id: int) -> None:
        """Remove a passenger from this node.

        Raises:
            KeyError: If passenger is not at this node.
        """
        self.passenger_ids.remove(passenger_id)

    def clone(self) -> NodeState:
        """Create a deep copy of this node state."""
        return NodeState(
            node_id=self.node_id,
            building_slots=[slot.clone() for slot in self.building_slots],
            passenger_ids=set(self.passenger_ids),
            is_train_station=self.is_train_station,
            is_central_park=self.is_central_park,
            position=self.position
        )


@dataclass
class RailSegment:
    """A single rail segment owned by a player on an edge."""

    player_id: int


@dataclass
class EdgeState:
    """State of a single edge (street) on the board.

    Attributes:
        edge_id: Canonical edge identifier (smaller_node_id, larger_node_id).
        rail_segments: List of rail segments on this edge (one per player max).
    """

    edge_id: EdgeId
    rail_segments: list[RailSegment] = field(default_factory=list)

    @property
    def endpoints(self) -> tuple[NodeId, NodeId]:
        """Return the two node IDs connected by this edge."""
        return self.edge_id

    def get_player_ids(self) -> set[int]:
        """Return the set of player IDs that have rails on this edge."""
        return {segment.player_id for segment in self.rail_segments}

    def has_player_rail(self, player_id: int) -> bool:
        """Check if a specific player has a rail segment on this edge."""
        return player_id in self.get_player_ids()

    def is_empty(self) -> bool:
        """Check if this edge has no rail segments."""
        return len(self.rail_segments) == 0

    def add_rail(self, player_id: int) -> None:
        """Add a rail segment for a player.

        Raises:
            ValueError: If player already has a rail on this edge.
        """
        if self.has_player_rail(player_id):
            raise ValueError(f"Player {player_id} already has a rail on this edge")
        self.rail_segments.append(RailSegment(player_id=player_id))

    def clone(self) -> EdgeState:
        """Create a deep copy of this edge state."""
        return EdgeState(
            edge_id=self.edge_id,
            rail_segments=[RailSegment(player_id=s.player_id) for s in self.rail_segments]
        )


@dataclass
class BoardGraph:
    """The game board represented as an attributed graph.

    The topology (nodes, edges, adjacency) is fixed at creation.
    Only the occupancy state (buildings, passengers, rails) changes during play.

    Attributes:
        nodes: Mapping from node ID to node state.
        edges: Mapping from edge ID to edge state.
        adjacency: Mapping from node ID to set of adjacent node IDs.
    """

    nodes: dict[NodeId, NodeState] = field(default_factory=dict)
    edges: dict[EdgeId, EdgeState] = field(default_factory=dict)
    adjacency: dict[NodeId, set[NodeId]] = field(default_factory=dict)

    def get_node(self, node_id: NodeId) -> NodeState:
        """Get the state of a specific node.

        Raises:
            KeyError: If node does not exist.
        """
        return self.nodes[node_id]

    def get_edge(self, node_a: NodeId, node_b: NodeId) -> EdgeState:
        """Get the state of the edge between two nodes.

        Raises:
            KeyError: If edge does not exist.
        """
        edge_id = make_edge_id(node_a, node_b)
        return self.edges[edge_id]

    def get_neighbors(self, node_id: NodeId) -> set[NodeId]:
        """Get all nodes adjacent to a given node."""
        return self.adjacency.get(node_id, set())

    def get_train_stations(self) -> list[NodeState]:
        """Return all train station nodes."""
        return [node for node in self.nodes.values() if node.is_train_station]

    def get_central_parks(self) -> list[NodeState]:
        """Return all central park nodes."""
        return [node for node in self.nodes.values() if node.is_central_park]

    def get_player_edges(self, player_id: int) -> list[EdgeState]:
        """Return all edges where a player has rail segments."""
        return [
            edge for edge in self.edges.values()
            if edge.has_player_rail(player_id)
        ]

    def get_player_network_nodes(self, player_id: int) -> set[NodeId]:
        """Return all nodes connected by a player's rail network."""
        player_edges = self.get_player_edges(player_id)
        nodes: set[NodeId] = set()
        for edge in player_edges:
            nodes.add(edge.edge_id[0])
            nodes.add(edge.edge_id[1])
        return nodes

    def get_player_network_endpoints(self, player_id: int) -> set[NodeId]:
        """Return endpoint nodes of a player's rail network.

        An endpoint is a node where the player's rail network terminates,
        meaning the player has exactly one connected edge at that node.

        Special case: If a player's network forms a loop (no endpoints),
        all nodes in the network are considered endpoints for expansion purposes.
        """
        # Count how many of the player's edges touch each node
        node_edge_count: dict[NodeId, int] = {}
        player_edges = self.get_player_edges(player_id)

        for edge in player_edges:
            for node_id in edge.edge_id:
                node_edge_count[node_id] = node_edge_count.get(node_id, 0) + 1

        # Endpoints are nodes with exactly one connected edge
        endpoints = {
            node_id for node_id, count in node_edge_count.items()
            if count == 1
        }

        # If network forms a complete loop (no endpoints), all nodes are endpoints
        if not endpoints and node_edge_count:
            endpoints = set(node_edge_count.keys())

        return endpoints

    def get_reachable_nodes(self, player_id: int, start_node: NodeId) -> set[NodeId]:
        """Return all nodes reachable from start_node via player's rail network.

        Uses breadth-first search to find all connected nodes.
        """
        if start_node not in self.nodes:
            return set()

        visited: set[NodeId] = set()
        queue: list[NodeId] = [start_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Check all adjacent nodes
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                # Only traverse if player has rail on this edge
                edge = self.get_edge(current, neighbor)
                if edge.has_player_rail(player_id):
                    queue.append(neighbor)

        return visited

    def get_empty_edges_at_node(self, node_id: NodeId) -> list[EdgeState]:
        """Return all edges adjacent to a node that have no rail segments."""
        empty_edges: list[EdgeState] = []
        for neighbor in self.get_neighbors(node_id):
            edge = self.get_edge(node_id, neighbor)
            if edge.is_empty():
                empty_edges.append(edge)
        return empty_edges

    def get_all_empty_slots_by_zone(self, zone: Zone) -> list[tuple[NodeId, BuildingSlot]]:
        """Return all empty building slots of a specific zone across all nodes.

        Returns list of (node_id, slot) tuples.
        """
        result: list[tuple[NodeId, BuildingSlot]] = []
        for node_id, node in self.nodes.items():
            for slot in node.get_empty_slots_by_zone(zone):
                result.append((node_id, slot))
        return result

    def get_empty_slots(self) -> dict[Zone, list[tuple[NodeId, BuildingSlot]]]:
        """Return all empty building slots grouped by zone.

        Returns a dictionary mapping each Zone to a list of (node_id, slot) tuples.
        """
        result: dict[Zone, list[tuple[NodeId, BuildingSlot]]] = {
            Zone.A: [],
            Zone.B: [],
            Zone.C: [],
            Zone.D: [],
        }
        for node_id, node in self.nodes.items():
            for slot in node.get_empty_slots():
                result[slot.zone].append((node_id, slot))
        return result

    def has_empty_slots_in_zone(self, zone: Zone) -> bool:
        """Check if any empty building slots exist in the given zone."""
        return any(
            node.get_empty_slots_by_zone(zone)
            for node in self.nodes.values()
        )

    def get_nodes_with_building_type(
        self, building_type: BuildingType
    ) -> list[NodeState]:
        """Return all nodes that have at least one building of the given type."""
        return [
            node for node in self.nodes.values()
            if node.has_building_type(building_type)
        ]

    def get_building_count(self, building_type: BuildingType) -> int:
        """Return the total number of buildings of the given type on the board."""
        count = 0
        for node in self.nodes.values():
            for slot in node.building_slots:
                if slot.building == building_type:
                    count += 1
        return count

    def clone(self) -> BoardGraph:
        """Create a deep copy of this board graph."""
        new_board = BoardGraph()
        new_board.nodes = {node_id: node.clone() for node_id, node in self.nodes.items()}
        new_board.edges = {edge_id: edge.clone() for edge_id, edge in self.edges.items()}
        new_board.adjacency = {node_id: set(neighbors) for node_id, neighbors in self.adjacency.items()}
        return new_board
