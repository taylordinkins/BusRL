"""Board data loader for the Bus game engine.

Loads and validates board topology from JSON files, converting them
into BoardGraph instances ready for use in the game.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import sys
import os

from core.constants import Zone
from core.board import (
    BoardGraph,
    NodeState,
    EdgeState,
    BuildingSlot,
    make_edge_id,
)


def resource_path(relative_path: str) -> Path:
    """
    Get absolute path to a resource, works for dev and PyInstaller exe.
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller bundles resources in a temporary folder
        return Path(sys._MEIPASS) / "data" / relative_path
    
    # Dev mode: look relative to this file (in the data/ directory)
    return Path(__file__).parent / relative_path

class BoardLoadError(Exception):
    """Raised when board loading or validation fails."""
    pass


class BoardLoader:
    """Loads and validates board data from JSON files."""

    # Expected counts for the standard Bus board
    EXPECTED_TRAIN_STATIONS = 2
    EXPECTED_CENTRAL_PARKS = 4

    def __init__(self, strict: bool = True):
        """Initialize the loader.

        Args:
            strict: If True, enforce strict validation (train stations,
                    central parks counts). Set to False for custom boards.
        """
        self.strict = strict

    def load_from_file(self, file_path: str | Path) -> BoardGraph:
        """Load a board from a JSON file.

        Args:
            file_path: Path to the JSON board file.

        Returns:
            A BoardGraph instance with the loaded topology.

        Raises:
            BoardLoadError: If the file cannot be read or parsed.
            BoardLoadError: If validation fails.
        """
        path = Path(file_path)

        if not path.exists():
            raise BoardLoadError(f"Board file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise BoardLoadError(f"Invalid JSON in board file: {e}")
        except IOError as e:
            raise BoardLoadError(f"Error reading board file: {e}")

        return self.load_from_dict(data)

    def load_from_dict(self, data: dict[str, Any]) -> BoardGraph:
        """Load a board from a dictionary.

        Args:
            data: Dictionary containing 'nodes' and 'edges' keys.

        Returns:
            A BoardGraph instance with the loaded topology.

        Raises:
            BoardLoadError: If validation fails.
        """
        # Validate structure
        self._validate_structure(data)

        # Create the board graph
        graph = BoardGraph()

        # Load nodes
        node_ids = set()
        for node_data in data["nodes"]:
            node = self._create_node(node_data)
            if node.node_id in node_ids:
                raise BoardLoadError(f"Duplicate node ID: {node.node_id}")
            node_ids.add(node.node_id)
            graph.nodes[node.node_id] = node

        # Load edges and build adjacency
        for edge_data in data["edges"]:
            node_a, node_b = edge_data[0], edge_data[1]

            # Validate edge references valid nodes
            if node_a not in node_ids:
                raise BoardLoadError(f"Edge references unknown node: {node_a}")
            if node_b not in node_ids:
                raise BoardLoadError(f"Edge references unknown node: {node_b}")

            # Validate no self-loops
            if node_a == node_b:
                raise BoardLoadError(f"Self-loop edge not allowed: [{node_a}, {node_b}]")

            # Create canonical edge ID
            edge_id = make_edge_id(node_a, node_b)

            # Check for duplicate edges
            if edge_id in graph.edges:
                raise BoardLoadError(f"Duplicate edge: {edge_id}")

            # Create edge state
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)

            # Update adjacency (bidirectional)
            graph.adjacency.setdefault(node_a, set()).add(node_b)
            graph.adjacency.setdefault(node_b, set()).add(node_a)

        # Ensure all nodes have adjacency entries (even isolated nodes)
        for node_id in node_ids:
            graph.adjacency.setdefault(node_id, set())

        # Validate the complete graph
        self._validate_graph(graph)

        return graph

    def _validate_structure(self, data: dict[str, Any]) -> None:
        """Validate the basic structure of the board data."""
        if not isinstance(data, dict):
            raise BoardLoadError("Board data must be a dictionary")

        if "nodes" not in data:
            raise BoardLoadError("Board data missing 'nodes' key")

        if "edges" not in data:
            raise BoardLoadError("Board data missing 'edges' key")

        if not isinstance(data["nodes"], list):
            raise BoardLoadError("'nodes' must be a list")

        if not isinstance(data["edges"], list):
            raise BoardLoadError("'edges' must be a list")

        if len(data["nodes"]) == 0:
            raise BoardLoadError("Board must have at least one node")

    def _create_node(self, node_data: dict[str, Any]) -> NodeState:
        """Create a NodeState from node data dictionary."""
        # Validate required fields
        required_fields = ["id", "building_slots", "is_train_station", "is_central_park", "position"]
        for field in required_fields:
            if field not in node_data:
                raise BoardLoadError(f"Node missing required field: {field}")

        node_id = node_data["id"]
        if not isinstance(node_id, int) or node_id < 0:
            raise BoardLoadError(f"Invalid node ID: {node_id}")

        # Parse building slots
        building_slots = []
        for zone_str in node_data["building_slots"]:
            try:
                zone = Zone(zone_str)
            except ValueError:
                raise BoardLoadError(
                    f"Invalid zone '{zone_str}' in node {node_id}. "
                    f"Valid zones: A, B, C, D"
                )
            building_slots.append(BuildingSlot(zone=zone))

        # Parse position
        position_data = node_data["position"]
        if not isinstance(position_data, dict) or "x" not in position_data or "y" not in position_data:
            raise BoardLoadError(f"Invalid position format for node {node_id}")
        position = (position_data["x"], position_data["y"])

        # Validate train station / central park constraints
        is_train_station = node_data["is_train_station"]
        is_central_park = node_data["is_central_park"]

        if is_train_station and is_central_park:
            raise BoardLoadError(
                f"Node {node_id} cannot be both train station and central park"
            )

        if is_train_station and building_slots:
            raise BoardLoadError(
                f"Train station (node {node_id}) should not have building slots"
            )

        return NodeState(
            node_id=node_id,
            building_slots=building_slots,
            is_train_station=is_train_station,
            is_central_park=is_central_park,
            position=position,
        )

    def _validate_graph(self, graph: BoardGraph) -> None:
        """Validate the complete graph structure."""
        if self.strict:
            # Check train station count
            train_stations = graph.get_train_stations()
            if len(train_stations) != self.EXPECTED_TRAIN_STATIONS:
                raise BoardLoadError(
                    f"Expected {self.EXPECTED_TRAIN_STATIONS} train stations, "
                    f"found {len(train_stations)}"
                )

            # Check central park count
            central_parks = graph.get_central_parks()
            if len(central_parks) != self.EXPECTED_CENTRAL_PARKS:
                raise BoardLoadError(
                    f"Expected {self.EXPECTED_CENTRAL_PARKS} central parks, "
                    f"found {len(central_parks)}"
                )

        # Check graph connectivity (all nodes should be reachable from any node)
        if len(graph.nodes) > 1:
            start_node = next(iter(graph.nodes.keys()))
            visited = set()
            self._dfs(graph, start_node, visited)

            if len(visited) != len(graph.nodes):
                unreachable = set(graph.nodes.keys()) - visited
                raise BoardLoadError(
                    f"Graph is not connected. Unreachable nodes: {unreachable}"
                )

    def _dfs(self, graph: BoardGraph, node_id: int, visited: set[int]) -> None:
        """Depth-first search to check connectivity."""
        visited.add(node_id)
        for neighbor in graph.adjacency.get(node_id, set()):
            if neighbor not in visited:
                self._dfs(graph, neighbor, visited)


def load_board(file_path: str | Path, strict: bool = True) -> BoardGraph:
    """Convenience function to load a board from a file.

    Args:
        file_path: Path to the JSON board file.
        strict: If True, enforce strict validation.

    Returns:
        A BoardGraph instance with the loaded topology.
    """
    loader = BoardLoader(strict=strict)
    return loader.load_from_file(file_path)


def load_default_board() -> BoardGraph:
    """Load the default Bus board.

    Returns:
        A BoardGraph instance with the standard Bus board topology.

    Raises:
        BoardLoadError: If the default board file is missing or invalid.
    """
    #default_path = Path(__file__).parent / "default_board.json"
    default_path = resource_path("default_board.json")
    return load_board(default_path, strict=True)


def get_board_stats(graph: BoardGraph) -> dict[str, Any]:
    """Get statistics about a board graph.

    Args:
        graph: The board graph to analyze.

    Returns:
        Dictionary with board statistics.
    """
    # Count building slots by zone
    zone_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_slots = 0

    for node in graph.nodes.values():
        for slot in node.building_slots:
            zone_counts[slot.zone.value] += 1
            total_slots += 1

    return {
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "num_train_stations": len(graph.get_train_stations()),
        "num_central_parks": len(graph.get_central_parks()),
        "total_building_slots": total_slots,
        "building_slots_by_zone": zone_counts,
    }
