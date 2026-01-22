"""Tests for the data loading module and integration with core components."""

import json
import tempfile
from pathlib import Path

import pytest

from data.loader import (
    BoardLoader,
    BoardLoadError,
    load_board,
    load_default_board,
    get_board_stats,
)
from core.constants import Zone, Phase, BuildingType, ActionAreaType
from core.board import BoardGraph, NodeState, EdgeState, BuildingSlot, make_edge_id
from core.player import Player
from core.components import PassengerManager
from core.game_state import GameState


# =============================================================================
# BoardLoader Tests
# =============================================================================

class TestBoardLoader:
    """Test BoardLoader class."""

    def test_load_minimal_valid_board(self):
        """Should load a minimal valid board."""
        data = {
            "nodes": [
                {
                    "id": 0,
                    "building_slots": ["A"],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 0, "y": 0}
                },
                {
                    "id": 1,
                    "building_slots": ["B"],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 1, "y": 0}
                }
            ],
            "edges": [[0, 1]]
        }

        loader = BoardLoader(strict=False)
        graph = loader.load_from_dict(data)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.get_node(0).building_slots[0].zone == Zone.A
        assert graph.get_node(1).building_slots[0].zone == Zone.B

    def test_load_board_with_all_zones(self):
        """Should load nodes with all zone types."""
        data = {
            "nodes": [
                {
                    "id": 0,
                    "building_slots": ["A", "B", "C", "D"],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 0, "y": 0}
                },
                {
                    "id": 1,
                    "building_slots": [],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 1, "y": 0}
                }
            ],
            "edges": [[0, 1]]
        }

        loader = BoardLoader(strict=False)
        graph = loader.load_from_dict(data)

        node = graph.get_node(0)
        zones = [slot.zone for slot in node.building_slots]
        assert zones == [Zone.A, Zone.B, Zone.C, Zone.D]

    def test_load_train_station(self):
        """Should load train station nodes."""
        data = {
            "nodes": [
                {
                    "id": 0,
                    "building_slots": [],
                    "is_train_station": True,
                    "is_central_park": False,
                    "position": {"x": 0, "y": 0}
                },
                {
                    "id": 1,
                    "building_slots": ["A"],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 1, "y": 0}
                }
            ],
            "edges": [[0, 1]]
        }

        loader = BoardLoader(strict=False)
        graph = loader.load_from_dict(data)

        stations = graph.get_train_stations()
        assert len(stations) == 1
        assert stations[0].node_id == 0

    def test_load_central_park(self):
        """Should load central park nodes."""
        data = {
            "nodes": [
                {
                    "id": 0,
                    "building_slots": ["A", "A"],
                    "is_train_station": False,
                    "is_central_park": True,
                    "position": {"x": 0, "y": 0}
                },
                {
                    "id": 1,
                    "building_slots": ["B"],
                    "is_train_station": False,
                    "is_central_park": False,
                    "position": {"x": 1, "y": 0}
                }
            ],
            "edges": [[0, 1]]
        }

        loader = BoardLoader(strict=False)
        graph = loader.load_from_dict(data)

        parks = graph.get_central_parks()
        assert len(parks) == 1
        assert parks[0].node_id == 0

    def test_adjacency_built_correctly(self):
        """Should build bidirectional adjacency."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
                {"id": 2, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 2, "y": 0}},
            ],
            "edges": [[0, 1], [1, 2]]
        }

        loader = BoardLoader(strict=False)
        graph = loader.load_from_dict(data)

        assert graph.get_neighbors(0) == {1}
        assert graph.get_neighbors(1) == {0, 2}
        assert graph.get_neighbors(2) == {1}


class TestBoardLoaderValidation:
    """Test BoardLoader validation."""

    def test_missing_nodes_key(self):
        """Should reject data without 'nodes' key."""
        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="missing 'nodes'"):
            loader.load_from_dict({"edges": []})

    def test_missing_edges_key(self):
        """Should reject data without 'edges' key."""
        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="missing 'edges'"):
            loader.load_from_dict({"nodes": []})

    def test_empty_nodes(self):
        """Should reject board with no nodes."""
        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="at least one node"):
            loader.load_from_dict({"nodes": [], "edges": []})

    def test_duplicate_node_id(self):
        """Should reject duplicate node IDs."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
            ],
            "edges": []
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="Duplicate node ID"):
            loader.load_from_dict(data)

    def test_invalid_zone(self):
        """Should reject invalid zone letters."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": ["X"], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
            ],
            "edges": []
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="Invalid zone"):
            loader.load_from_dict(data)

    def test_edge_unknown_node(self):
        """Should reject edges referencing unknown nodes."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
            ],
            "edges": [[0, 99]]
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="unknown node"):
            loader.load_from_dict(data)

    def test_self_loop_edge(self):
        """Should reject self-loop edges."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
            ],
            "edges": [[0, 0]]
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="Self-loop"):
            loader.load_from_dict(data)

    def test_duplicate_edge(self):
        """Should reject duplicate edges."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
            ],
            "edges": [[0, 1], [1, 0]]  # Same edge, different order
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="Duplicate edge"):
            loader.load_from_dict(data)

    def test_train_station_with_building_slots(self):
        """Should reject train station with building slots."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": ["A"], "is_train_station": True,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
            ],
            "edges": []
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="should not have building slots"):
            loader.load_from_dict(data)

    def test_both_train_station_and_central_park(self):
        """Should reject node that is both train station and central park."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": True,
                 "is_central_park": True, "position": {"x": 0, "y": 0}},
            ],
            "edges": []
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="cannot be both"):
            loader.load_from_dict(data)

    def test_disconnected_graph(self):
        """Should reject disconnected graphs."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
                {"id": 2, "building_slots": [], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 2, "y": 0}},
            ],
            "edges": [[0, 1]]  # Node 2 is disconnected
        }

        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="not connected"):
            loader.load_from_dict(data)


class TestBoardLoaderStrictMode:
    """Test BoardLoader strict validation mode."""

    def test_strict_mode_requires_train_stations(self):
        """Strict mode should require exactly 2 train stations."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": True,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": ["A"], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
            ],
            "edges": [[0, 1]]
        }

        loader = BoardLoader(strict=True)
        with pytest.raises(BoardLoadError, match="train stations"):
            loader.load_from_dict(data)

    def test_strict_mode_requires_central_parks(self):
        """Strict mode should require exactly 4 central parks."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": [], "is_train_station": True,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": [], "is_train_station": True,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
                {"id": 2, "building_slots": ["A"], "is_train_station": False,
                 "is_central_park": True, "position": {"x": 2, "y": 0}},
            ],
            "edges": [[0, 1], [1, 2]]
        }

        loader = BoardLoader(strict=True)
        with pytest.raises(BoardLoadError, match="central parks"):
            loader.load_from_dict(data)


class TestBoardLoaderFile:
    """Test BoardLoader file operations."""

    def test_load_from_file(self):
        """Should load board from JSON file."""
        data = {
            "nodes": [
                {"id": 0, "building_slots": ["A"], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 0, "y": 0}},
                {"id": 1, "building_slots": ["B"], "is_train_station": False,
                 "is_central_park": False, "position": {"x": 1, "y": 0}},
            ],
            "edges": [[0, 1]]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = BoardLoader(strict=False)
            graph = loader.load_from_file(temp_path)
            assert len(graph.nodes) == 2
        finally:
            Path(temp_path).unlink()

    def test_load_missing_file(self):
        """Should raise error for missing file."""
        loader = BoardLoader(strict=False)
        with pytest.raises(BoardLoadError, match="not found"):
            loader.load_from_file("/nonexistent/path.json")

    def test_load_invalid_json(self):
        """Should raise error for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            loader = BoardLoader(strict=False)
            with pytest.raises(BoardLoadError, match="Invalid JSON"):
                loader.load_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Default Board Tests
# =============================================================================

class TestDefaultBoard:
    """Test loading the default Bus board."""

    def test_load_default_board(self):
        """Should load the default board successfully."""
        graph = load_default_board()

        assert graph is not None
        assert len(graph.nodes) == 36
        assert len(graph.edges) == 70

    def test_default_board_train_stations(self):
        """Default board should have 2 train stations."""
        graph = load_default_board()
        stations = graph.get_train_stations()

        assert len(stations) == 2
        # Train stations should have no building slots
        for station in stations:
            assert len(station.building_slots) == 0

    def test_default_board_central_parks(self):
        """Default board should have 4 central parks."""
        graph = load_default_board()
        parks = graph.get_central_parks()

        assert len(parks) == 4
        # Central parks have zone A building slots
        for park in parks:
            assert all(slot.zone == Zone.A for slot in park.building_slots)

    def test_default_board_stats(self):
        """Should get correct statistics for default board."""
        graph = load_default_board()
        stats = get_board_stats(graph)

        assert stats["num_nodes"] == 36
        assert stats["num_edges"] == 70
        assert stats["num_train_stations"] == 2
        assert stats["num_central_parks"] == 4
        assert stats["total_building_slots"] > 0

    def test_default_board_connectivity(self):
        """All nodes in default board should be connected."""
        graph = load_default_board()

        # Try to reach all nodes from node 0
        visited = set()
        to_visit = [0]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    to_visit.append(neighbor)

        assert visited == set(graph.nodes.keys())


# =============================================================================
# Integration Tests: Board + Core Components
# =============================================================================

class TestBoardCoreIntegration:
    """Integration tests for loaded board with core components."""

    @pytest.fixture
    def loaded_board(self) -> BoardGraph:
        """Load the default board for testing."""
        return load_default_board()

    def test_create_game_state_with_loaded_board(self, loaded_board: BoardGraph):
        """Should create GameState from loaded board."""
        state = GameState.create_initial_state(loaded_board, num_players=3)

        assert state.board is loaded_board
        assert len(state.players) == 3
        assert state.phase == Phase.SETUP_BUILDINGS

    def test_place_building_on_loaded_board(self, loaded_board: BoardGraph):
        """Should place buildings on slots from loaded board."""
        # Find a node with zone A slots
        parks = loaded_board.get_central_parks()
        assert len(parks) > 0

        park = parks[0]
        empty_slots = park.get_empty_slots_by_zone(Zone.A)
        assert len(empty_slots) > 0

        # Place a building
        slot = empty_slots[0]
        slot.place_building(BuildingType.HOUSE)

        assert slot.building == BuildingType.HOUSE
        assert not slot.is_empty()

    def test_place_rails_on_loaded_board(self, loaded_board: BoardGraph):
        """Should place rails on edges from loaded board."""
        # Get an edge
        edge_id = next(iter(loaded_board.edges.keys()))
        edge = loaded_board.edges[edge_id]

        assert edge.is_empty()

        # Place rail for player 0
        edge.add_rail(player_id=0)

        assert not edge.is_empty()
        assert edge.has_player_rail(0)

    def test_passenger_spawning_at_stations(self, loaded_board: BoardGraph):
        """Should spawn passengers at train stations."""
        state = GameState.create_initial_state(loaded_board, num_players=3)

        stations = loaded_board.get_train_stations()
        assert len(stations) == 2

        # Spawn passengers at first station
        station = stations[0]
        for _ in range(3):
            p = state.passenger_manager.create_passenger(location=station.node_id)
            station.add_passenger(p.passenger_id)

        assert len(station.passenger_ids) == 3
        assert state.passenger_manager.count() == 3

    def test_passenger_initial_placement_at_parks(self, loaded_board: BoardGraph):
        """Should place initial passengers at central parks."""
        state = GameState.create_initial_state(loaded_board, num_players=3)

        parks = loaded_board.get_central_parks()
        assert len(parks) == 4

        # Place 1 passenger at each park (initial game setup)
        for park in parks:
            p = state.passenger_manager.create_passenger(location=park.node_id)
            park.add_passenger(p.passenger_id)

        assert state.passenger_manager.count() == 4

        # Verify distribution
        for park in parks:
            assert len(park.passenger_ids) == 1

    def test_rail_network_traversal(self, loaded_board: BoardGraph):
        """Should traverse rail network on loaded board."""
        # Build a small network from node 0
        node_0_neighbors = loaded_board.get_neighbors(0)
        assert len(node_0_neighbors) > 0

        first_neighbor = next(iter(node_0_neighbors))
        edge_id = make_edge_id(0, first_neighbor)

        # Place rail
        loaded_board.edges[edge_id].add_rail(player_id=0)

        # Check reachability
        reachable = loaded_board.get_reachable_nodes(0, start_node=0)
        assert 0 in reachable
        assert first_neighbor in reachable

    def test_building_slots_by_zone(self, loaded_board: BoardGraph):
        """Should correctly categorize building slots by zone."""
        all_slots = loaded_board.get_empty_slots()

        # All zones should be present
        assert Zone.A in all_slots
        assert Zone.B in all_slots
        assert Zone.C in all_slots
        assert Zone.D in all_slots

        # Zone A is innermost, should have slots (central parks)
        assert len(all_slots[Zone.A]) > 0


class TestFullGameSetupIntegration:
    """Integration tests simulating full game setup with loaded board."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a game state with the default board."""
        board = load_default_board()
        return GameState.create_initial_state(board, num_players=4)

    def test_setup_initial_passengers(self, game_state: GameState):
        """Should set up initial passengers at central parks."""
        parks = game_state.board.get_central_parks()

        # Place initial passenger at each park
        for park in parks:
            p = game_state.passenger_manager.create_passenger(location=park.node_id)
            park.add_passenger(p.passenger_id)

        assert game_state.passenger_manager.count() == 4

        # Validate state
        errors = game_state.validate()
        assert len(errors) == 0

    def test_setup_phase_building_placement(self, game_state: GameState):
        """Should allow building placement in setup phase."""
        assert game_state.phase == Phase.SETUP_BUILDINGS

        # Find zone A slots (innermost, must be filled first)
        zone_a_slots = game_state.board.get_all_empty_slots_by_zone(Zone.A)

        # First player places 2 buildings
        player = game_state.get_current_player()
        assert player.player_id == 0

        # Place two buildings in zone A
        for i, (node_id, slot) in enumerate(zone_a_slots[:2]):
            slot.place_building(BuildingType.HOUSE if i == 0 else BuildingType.OFFICE)

        # Verify placements
        placed_count = 0
        for node_id, slot in zone_a_slots[:2]:
            if not slot.is_empty():
                placed_count += 1
        assert placed_count == 2

    def test_action_board_with_loaded_board(self, game_state: GameState):
        """Action board should work with game state from loaded board."""
        game_state.set_phase(Phase.CHOOSING_ACTIONS)

        # Players place markers
        for i, area in enumerate([ActionAreaType.LINE_EXPANSION, ActionAreaType.BUSES, ActionAreaType.VRROOMM]):
            player = game_state.players[i % len(game_state.players)]
            player.place_marker()
            game_state.action_board.place_marker(area, player.player_id)

        assert game_state.action_board.count_total_markers() == 3

    def test_clone_with_loaded_board(self, game_state: GameState):
        """Should clone game state with loaded board."""
        # Modify state
        game_state.players[0].add_score(10)
        parks = game_state.board.get_central_parks()
        p = game_state.passenger_manager.create_passenger(location=parks[0].node_id)
        parks[0].add_passenger(p.passenger_id)

        # Clone
        clone = game_state.clone()

        # Verify independence
        game_state.players[0].add_score(5)
        assert clone.players[0].score == 10

        # Board should be cloned
        assert clone.board is not game_state.board
        assert len(clone.board.nodes) == len(game_state.board.nodes)

    def test_serialization_with_loaded_board(self, game_state: GameState):
        """Should serialize game state with loaded board."""
        # Add some state
        game_state.players[0].add_score(5)
        game_state.action_board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)

        # Serialize
        data = game_state.to_dict()

        assert data["phase"] == "setup_buildings"
        assert len(data["players"]) == 4
        assert len(data["board_state"]["nodes"]) == 36


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
