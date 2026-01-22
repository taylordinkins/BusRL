"""Tests for board visualization with integration testing of core components.

These tests verify:
1. Visualization of the default board (before setup)
2. Visualization after simulating game setup (buildings, rails, passengers)
3. Integration with all core components
"""

import tempfile
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from data.loader import load_default_board
from data.graph_vis import BoardVisualizer, visualize_board, visualize_default_board
from core.board import BoardGraph, NodeState, EdgeState, BuildingSlot, make_edge_id
from core.constants import Zone, BuildingType, Phase
from core.player import Player
from core.components import PassengerManager
from core.game_state import GameState


# =============================================================================
# Basic Visualization Tests
# =============================================================================

class TestBoardVisualizer:
    """Test BoardVisualizer class."""

    @pytest.fixture
    def simple_board(self) -> BoardGraph:
        """Create a simple test board for visualization."""
        graph = BoardGraph()

        # Create nodes in a grid pattern
        positions = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        ]

        for i, pos in enumerate(positions):
            is_station = (i == 0)
            is_park = (i == 4)
            slots = []
            if not is_station:
                zone = [Zone.A, Zone.B, Zone.C, Zone.D][i % 4]
                slots = [BuildingSlot(zone=zone)]

            graph.nodes[i] = NodeState(
                node_id=i,
                building_slots=slots,
                is_train_station=is_station,
                is_central_park=is_park,
                position=pos,
            )

        # Create edges in a grid
        edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
        for a, b in edges:
            edge_id = make_edge_id(a, b)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(a, set()).add(b)
            graph.adjacency.setdefault(b, set()).add(a)

        return graph

    def test_visualizer_creation(self, simple_board: BoardGraph):
        """Should create visualizer from board."""
        vis = BoardVisualizer(simple_board)
        assert vis.board is simple_board
        assert vis.figsize == (14, 12)

    def test_visualize_returns_figure(self, simple_board: BoardGraph):
        """Should return matplotlib Figure."""
        vis = BoardVisualizer(simple_board)
        fig = vis.visualize(show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_with_custom_title(self, simple_board: BoardGraph):
        """Should use custom title."""
        vis = BoardVisualizer(simple_board)
        fig = vis.visualize(title="Test Board", show=False)

        # Check title is set (via axes)
        ax = fig.axes[0]
        assert ax.get_title() == "Test Board"
        plt.close(fig)

    def test_visualize_save_to_file(self, simple_board: BoardGraph):
        """Should save visualization to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            vis = BoardVisualizer(simple_board)
            fig = vis.visualize(save_path=temp_path, show=False)

            assert temp_path.exists()
            assert temp_path.stat().st_size > 0
            plt.close(fig)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_visualize_without_legend(self, simple_board: BoardGraph):
        """Should work without legend."""
        vis = BoardVisualizer(simple_board)
        fig = vis.visualize(show_legend=False, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_without_buildings(self, simple_board: BoardGraph):
        """Should work without building indicators."""
        vis = BoardVisualizer(simple_board)
        fig = vis.visualize(show_buildings=False, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_without_passengers(self, simple_board: BoardGraph):
        """Should work without passenger indicators."""
        vis = BoardVisualizer(simple_board)
        fig = vis.visualize(show_passengers=False, show=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizeFunctions:
    """Test convenience visualization functions."""

    @pytest.fixture
    def simple_board(self) -> BoardGraph:
        """Create a simple test board."""
        graph = BoardGraph()
        for i in range(3):
            graph.nodes[i] = NodeState(
                node_id=i,
                building_slots=[BuildingSlot(zone=Zone.A)],
                position=(i, 0),
            )
        for i in range(2):
            edge_id = make_edge_id(i, i + 1)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(i, set()).add(i + 1)
            graph.adjacency.setdefault(i + 1, set()).add(i)
        return graph

    def test_visualize_board_function(self, simple_board: BoardGraph):
        """Should visualize board via convenience function."""
        fig = visualize_board(simple_board, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_default_board_function(self):
        """Should visualize default board."""
        fig = visualize_default_board(show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Default Board Visualization Tests
# =============================================================================

class TestDefaultBoardVisualization:
    """Test visualization of the default Bus board."""

    @pytest.fixture
    def default_board(self) -> BoardGraph:
        """Load the default board."""
        return load_default_board()

    def test_visualize_default_board_before_setup(self, default_board: BoardGraph):
        """Should visualize default board in initial state."""
        fig = visualize_board(
            default_board,
            title="Default Board - Before Setup",
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_default_board_shows_all_nodes(self, default_board: BoardGraph):
        """Visualization should include all 36 nodes."""
        vis = BoardVisualizer(default_board)
        fig = vis.visualize(show=False)

        # The internal NetworkX graph should have all nodes
        G = vis._build_networkx_graph()
        assert len(G.nodes()) == 36
        plt.close(fig)

    def test_visualize_default_board_shows_all_edges(self, default_board: BoardGraph):
        """Visualization should include all 70 edges."""
        vis = BoardVisualizer(default_board)
        G = vis._build_networkx_graph()

        assert len(G.edges()) == 70

    def test_visualize_default_board_train_stations(self, default_board: BoardGraph):
        """Should identify train stations in visualization."""
        vis = BoardVisualizer(default_board)
        G = vis._build_networkx_graph()

        stations = [n for n, d in G.nodes(data=True) if d["is_train_station"]]
        assert len(stations) == 2

    def test_visualize_default_board_central_parks(self, default_board: BoardGraph):
        """Should identify central parks in visualization."""
        vis = BoardVisualizer(default_board)
        G = vis._build_networkx_graph()

        parks = [n for n, d in G.nodes(data=True) if d["is_central_park"]]
        assert len(parks) == 4

    def test_save_default_board_visualization(self, default_board: BoardGraph):
        """Should save default board visualization to file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            fig = visualize_board(
                default_board,
                title="Default Bus Board",
                save_path=temp_path,
                show=False,
            )

            assert temp_path.exists()
            # PNG file should be reasonably sized
            assert temp_path.stat().st_size > 10000
            plt.close(fig)
        finally:
            if temp_path.exists():
                temp_path.unlink()


# =============================================================================
# Integration Tests: Visualization with Game State
# =============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization with game components."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a game state with the default board."""
        board = load_default_board()
        return GameState.create_initial_state(board, num_players=4)

    def test_visualize_after_building_placement(self, game_state: GameState):
        """Should visualize board after placing buildings."""
        # Place some buildings in zone A (central parks have zone A slots)
        zone_a_slots = game_state.board.get_all_empty_slots_by_zone(Zone.A)

        # Place different building types
        building_types = [BuildingType.HOUSE, BuildingType.OFFICE, BuildingType.PUB]
        for i, (node_id, slot) in enumerate(zone_a_slots[:6]):
            slot.place_building(building_types[i % 3])

        # Visualize
        fig = visualize_board(
            game_state.board,
            title="Board After Building Placement",
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_after_rail_placement(self, game_state: GameState):
        """Should visualize board after placing rails."""
        board = game_state.board

        # Player 0 builds a rail network
        edges_to_build = [(0, 1), (1, 5), (5, 6)]
        for a, b in edges_to_build:
            edge_id = make_edge_id(a, b)
            if edge_id in board.edges:
                board.edges[edge_id].add_rail(player_id=0)
                game_state.players[0].place_rail()

        # Player 1 builds a different network
        edges_to_build_p1 = [(8, 9), (9, 17)]
        for a, b in edges_to_build_p1:
            edge_id = make_edge_id(a, b)
            if edge_id in board.edges:
                board.edges[edge_id].add_rail(player_id=1)
                game_state.players[1].place_rail()

        # Visualize
        fig = visualize_board(
            board,
            title="Board After Rail Placement (2 Players)",
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_shared_rails(self, game_state: GameState):
        """Should visualize edges with multiple player rails."""
        board = game_state.board

        # Multiple players share the same edge
        edge_id = make_edge_id(7, 8)
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=0)
            board.edges[edge_id].add_rail(player_id=1)
            board.edges[edge_id].add_rail(player_id=2)

        # Visualize
        fig = visualize_board(
            board,
            title="Board With Shared Rails",
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_with_passengers(self, game_state: GameState):
        """Should visualize board with passengers at nodes."""
        board = game_state.board
        pm = game_state.passenger_manager

        # Place passengers at central parks (initial setup)
        parks = board.get_central_parks()
        for park in parks:
            p = pm.create_passenger(location=park.node_id)
            park.add_passenger(p.passenger_id)

        # Place additional passengers at a train station
        stations = board.get_train_stations()
        if stations:
            station = stations[0]
            for _ in range(3):
                p = pm.create_passenger(location=station.node_id)
                station.add_passenger(p.passenger_id)

        # Visualize
        fig = visualize_board(
            board,
            title="Board With Passengers",
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualize_full_game_simulation(self, game_state: GameState):
        """Should visualize board after simulating partial game setup."""
        board = game_state.board
        pm = game_state.passenger_manager

        # 1. Place initial passengers at central parks
        parks = board.get_central_parks()
        for park in parks:
            p = pm.create_passenger(location=park.node_id)
            park.add_passenger(p.passenger_id)

        # 2. Place some buildings (simulating setup phase)
        zone_a_slots = board.get_all_empty_slots_by_zone(Zone.A)
        building_types = [BuildingType.HOUSE, BuildingType.OFFICE, BuildingType.PUB]
        for i, (node_id, slot) in enumerate(zone_a_slots[:8]):
            slot.place_building(building_types[i % 3])

        # 3. Place rails for multiple players
        # Player 0's network
        player_0_edges = [(0, 1), (1, 2), (2, 6), (6, 14)]
        for a, b in player_0_edges:
            edge_id = make_edge_id(a, b)
            if edge_id in board.edges and not board.edges[edge_id].has_player_rail(0):
                board.edges[edge_id].add_rail(player_id=0)

        # Player 1's network
        player_1_edges = [(8, 9), (9, 17), (8, 12), (12, 16)]
        for a, b in player_1_edges:
            edge_id = make_edge_id(a, b)
            if edge_id in board.edges and not board.edges[edge_id].has_player_rail(1):
                board.edges[edge_id].add_rail(player_id=1)

        # Player 2's network (shares some edges)
        player_2_edges = [(27, 30), (30, 26), (26, 22), (6, 14)]  # Shares (6,14) with P0
        for a, b in player_2_edges:
            edge_id = make_edge_id(a, b)
            if edge_id in board.edges and not board.edges[edge_id].has_player_rail(2):
                board.edges[edge_id].add_rail(player_id=2)

        # 4. Move some passengers (simulating Vrroomm)
        # Move a passenger from park to a node with a house
        if parks and len(pm.passengers) > 0:
            park = parks[0]
            if park.passenger_ids:
                p_id = next(iter(park.passenger_ids))
                # Find a node with a house
                house_nodes = board.get_nodes_with_building_type(BuildingType.HOUSE)
                if house_nodes:
                    dest = house_nodes[0]
                    park.remove_passenger(p_id)
                    pm.move_passenger(p_id, new_location=dest.node_id)
                    dest.add_passenger(p_id)

        # Visualize the full state
        fig = visualize_board(
            board,
            title="Bus Game - Mid-Game State Simulation",
            show=False,
        )

        assert isinstance(fig, plt.Figure)

        # Save to temp file to verify it renders correctly
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            fig.savefig(temp_path, dpi=100, bbox_inches="tight")
            assert temp_path.exists()
            assert temp_path.stat().st_size > 10000  # Should be a substantial image
        finally:
            plt.close(fig)
            if temp_path.exists():
                temp_path.unlink()

    def test_validate_state_after_visualization_changes(self, game_state: GameState):
        """State should remain valid after visualization-related changes."""
        board = game_state.board
        pm = game_state.passenger_manager

        # Make various changes
        parks = board.get_central_parks()
        for park in parks:
            p = pm.create_passenger(location=park.node_id)
            park.add_passenger(p.passenger_id)

        zone_a_slots = board.get_all_empty_slots_by_zone(Zone.A)
        for node_id, slot in zone_a_slots[:4]:
            slot.place_building(BuildingType.HOUSE)

        # Validate state
        errors = game_state.validate()
        assert len(errors) == 0

        # Visualize (should not affect state)
        fig = visualize_board(board, show=False)
        plt.close(fig)

        # State should still be valid
        errors = game_state.validate()
        assert len(errors) == 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
