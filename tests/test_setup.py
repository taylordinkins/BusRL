"""Tests for the game setup logic.

Tests cover:
1. Initial passenger placement at central parks
2. Building placement during SETUP_BUILDINGS phase
3. Forward rail placement during SETUP_RAILS_FORWARD phase
4. Reverse rail placement during SETUP_RAILS_REVERSE phase
5. Integration tests for complete setup flow
"""

import pytest

from core.constants import (
    Phase,
    Zone,
    BuildingType,
    INITIAL_PASSENGERS_AT_PARKS,
    TOTAL_RAIL_SEGMENTS,
)
from core.board import BoardGraph, NodeState, EdgeState, BuildingSlot, make_edge_id
from core.game_state import GameState
from engine.setup import (
    SetupManager,
    SetupAction,
    SetupValidationResult,
    initialize_game,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_board() -> BoardGraph:
    """Create a simple test board with:
    - 1 train station (node 0)
    - 2 central parks (nodes 1, 2)
    - 4 regular nodes with Zone A slots (nodes 3, 4, 5, 6)
    - Edges forming a connected graph
    """
    graph = BoardGraph()

    # Train station (no building slots)
    graph.nodes[0] = NodeState(
        node_id=0,
        is_train_station=True,
        position=(0, 0),
    )

    # Central parks with Zone A slots
    graph.nodes[1] = NodeState(
        node_id=1,
        is_central_park=True,
        building_slots=[BuildingSlot(zone=Zone.A)],
        position=(1, 0),
    )
    graph.nodes[2] = NodeState(
        node_id=2,
        is_central_park=True,
        building_slots=[BuildingSlot(zone=Zone.A)],
        position=(2, 0),
    )

    # Regular nodes with Zone A slots
    for i in range(3, 7):
        graph.nodes[i] = NodeState(
            node_id=i,
            building_slots=[BuildingSlot(zone=Zone.A)],
            position=(i, 0),
        )

    # Create edges: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 1-3, 2-4
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3), (2, 4)]
    for a, b in edges:
        edge_id = make_edge_id(a, b)
        graph.edges[edge_id] = EdgeState(edge_id=edge_id)
        graph.adjacency.setdefault(a, set()).add(b)
        graph.adjacency.setdefault(b, set()).add(a)

    return graph


@pytest.fixture
def game_state(simple_board: BoardGraph) -> GameState:
    """Create a game state for testing."""
    return GameState.create_initial_state(simple_board, num_players=3)


@pytest.fixture
def setup_manager(game_state: GameState) -> SetupManager:
    """Create a setup manager for testing."""
    return SetupManager(game_state)


# =============================================================================
# Initial Passenger Placement Tests
# =============================================================================


class TestInitialPassengerPlacement:
    """Test initial passenger placement at central parks."""

    def test_place_initial_passengers(self, game_state: GameState):
        """Should place one passenger at each central park."""
        manager = SetupManager(game_state)
        count = manager.place_initial_passengers()

        # Should place INITIAL_PASSENGERS_AT_PARKS per park
        parks = game_state.board.get_central_parks()
        expected_count = len(parks) * INITIAL_PASSENGERS_AT_PARKS
        assert count == expected_count

        # Each park should have passengers
        for park in parks:
            assert len(park.passenger_ids) == INITIAL_PASSENGERS_AT_PARKS

        # Passenger manager should track them
        assert game_state.passenger_manager.count() == expected_count

    def test_initialize_game_places_passengers(self, game_state: GameState):
        """initialize_game() should place initial passengers."""
        manager = initialize_game(game_state)

        parks = game_state.board.get_central_parks()
        for park in parks:
            assert len(park.passenger_ids) == INITIAL_PASSENGERS_AT_PARKS

    def test_passengers_have_correct_location(self, game_state: GameState):
        """Passengers should be tracked at their park locations."""
        manager = SetupManager(game_state)
        manager.place_initial_passengers()

        for passenger in game_state.passenger_manager.passengers.values():
            # Passenger's location should be a central park
            node = game_state.board.nodes[passenger.location]
            assert node.is_central_park
            # Node should contain this passenger
            assert passenger.passenger_id in node.passenger_ids


# =============================================================================
# Building Placement Tests
# =============================================================================


class TestBuildingPlacement:
    """Test building placement during SETUP_BUILDINGS phase."""

    def test_get_valid_building_slots(self, setup_manager: SetupManager):
        """Should return only Zone A slots."""
        valid_slots = setup_manager.get_valid_building_slots()

        # Should have slots (we have 6 nodes with Zone A slots)
        assert len(valid_slots) >= 6

        # All slots should be Zone A
        for node_id, slot_idx in valid_slots:
            node = setup_manager.state.board.nodes[node_id]
            slot = node.building_slots[slot_idx]
            assert slot.zone == Zone.A
            assert slot.is_empty()

    def test_validate_building_placement_success(self, setup_manager: SetupManager):
        """Should validate correct building placement."""
        valid_slots = setup_manager.get_valid_building_slots()
        node_id, slot_idx = valid_slots[0]

        result = setup_manager.validate_building_placement(
            player_id=0,
            node_id=node_id,
            slot_index=slot_idx,
            building_type=BuildingType.HOUSE,
        )

        assert result.valid
        assert result.reason is None

    def test_validate_building_placement_wrong_phase(self, setup_manager: SetupManager):
        """Should reject placement in wrong phase."""
        setup_manager.state.set_phase(Phase.CHOOSING_ACTIONS)

        result = setup_manager.validate_building_placement(
            player_id=0,
            node_id=1,
            slot_index=0,
            building_type=BuildingType.HOUSE,
        )

        assert not result.valid
        assert "SETUP_BUILDINGS" in result.reason

    def test_validate_building_placement_wrong_player(self, setup_manager: SetupManager):
        """Should reject placement when not player's turn."""
        # Player 0's turn, but player 1 tries to place
        result = setup_manager.validate_building_placement(
            player_id=1,
            node_id=1,
            slot_index=0,
            building_type=BuildingType.HOUSE,
        )

        assert not result.valid
        assert "turn" in result.reason.lower()

    def test_validate_building_placement_invalid_node(self, setup_manager: SetupManager):
        """Should reject placement on non-existent node."""
        result = setup_manager.validate_building_placement(
            player_id=0,
            node_id=999,
            slot_index=0,
            building_type=BuildingType.HOUSE,
        )

        assert not result.valid
        assert "does not exist" in result.reason

    def test_validate_building_placement_invalid_slot_index(
        self, setup_manager: SetupManager
    ):
        """Should reject invalid slot index."""
        result = setup_manager.validate_building_placement(
            player_id=0,
            node_id=1,
            slot_index=99,
            building_type=BuildingType.HOUSE,
        )

        assert not result.valid
        assert "Invalid slot index" in result.reason

    def test_place_building(self, setup_manager: SetupManager):
        """Should place building and track progress."""
        valid_slots = setup_manager.get_valid_building_slots()
        node_id, slot_idx = valid_slots[0]

        action = setup_manager.place_building(
            player_id=0,
            node_id=node_id,
            slot_index=slot_idx,
            building_type=BuildingType.HOUSE,
        )

        # Action should be recorded
        assert action.player_id == 0
        assert action.action_type == "building"
        assert action.details["building_type"] == "house"

        # Building should be placed
        slot = setup_manager.state.board.nodes[node_id].building_slots[slot_idx]
        assert slot.building == BuildingType.HOUSE

        # Progress should be tracked
        assert setup_manager._buildings_placed[0] == 1

    def test_place_building_invalid_raises(self, setup_manager: SetupManager):
        """Should raise ValueError for invalid placement."""
        with pytest.raises(ValueError):
            setup_manager.place_building(
                player_id=1,  # Wrong player
                node_id=1,
                slot_index=0,
                building_type=BuildingType.HOUSE,
            )

    def test_is_player_building_setup_complete(self, setup_manager: SetupManager):
        """Should track when player has placed all buildings."""
        assert not setup_manager.is_player_building_setup_complete(0)

        # Place first building
        valid_slots = setup_manager.get_valid_building_slots()
        setup_manager.place_building(0, valid_slots[0][0], valid_slots[0][1], BuildingType.HOUSE)
        assert not setup_manager.is_player_building_setup_complete(0)

        # Place second building
        valid_slots = setup_manager.get_valid_building_slots()
        setup_manager.place_building(0, valid_slots[0][0], valid_slots[0][1], BuildingType.OFFICE)
        assert setup_manager.is_player_building_setup_complete(0)

    def test_is_buildings_phase_complete(self, setup_manager: SetupManager):
        """Should detect when all players completed building placement."""
        assert not setup_manager.is_buildings_phase_complete()

        # Complete building placement for all players
        for player in setup_manager.state.players:
            setup_manager.state.global_state.current_player_idx = player.player_id
            for _ in range(SetupManager.BUILDINGS_PER_PLAYER):
                valid_slots = setup_manager.get_valid_building_slots()
                if valid_slots:
                    node_id, slot_idx = valid_slots[0]
                    setup_manager.place_building(
                        player.player_id, node_id, slot_idx, BuildingType.HOUSE
                    )

        assert setup_manager.is_buildings_phase_complete()


# =============================================================================
# Forward Rail Placement Tests
# =============================================================================


class TestForwardRailPlacement:
    """Test forward rail placement during SETUP_RAILS_FORWARD phase."""

    def test_get_valid_rail_edges_forward(self, setup_manager: SetupManager):
        """Should return all edges (forward allows any edge)."""
        valid_edges = setup_manager.get_valid_rail_edges_forward(player_id=0)

        # Should have all edges available
        assert len(valid_edges) == len(setup_manager.state.board.edges)

    def test_get_valid_rail_edges_forward_excludes_own_rails(
        self, setup_manager: SetupManager
    ):
        """Should exclude edges where player already has rail."""
        # Place a rail for player 0
        edge_id = (0, 1)
        setup_manager.state.board.edges[edge_id].add_rail(0)

        valid_edges = setup_manager.get_valid_rail_edges_forward(player_id=0)

        # Should exclude the edge with player's rail
        assert edge_id not in valid_edges
        assert len(valid_edges) == len(setup_manager.state.board.edges) - 1

    def test_validate_rail_placement_forward_success(self, setup_manager: SetupManager):
        """Should validate correct forward rail placement."""
        setup_manager.state.set_phase(Phase.SETUP_RAILS_FORWARD)

        result = setup_manager.validate_rail_placement_forward(
            player_id=0,
            edge_id=(0, 1),
        )

        assert result.valid

    def test_validate_rail_placement_forward_wrong_phase(
        self, setup_manager: SetupManager
    ):
        """Should reject placement in wrong phase."""
        result = setup_manager.validate_rail_placement_forward(
            player_id=0,
            edge_id=(0, 1),
        )

        assert not result.valid
        assert "SETUP_RAILS_FORWARD" in result.reason

    def test_validate_rail_placement_forward_allows_shared_edge(
        self, setup_manager: SetupManager
    ):
        """Forward setup should allow placing on edges with other players' rails."""
        setup_manager.state.set_phase(Phase.SETUP_RAILS_FORWARD)

        # Player 1 already has rail on this edge
        edge_id = (0, 1)
        setup_manager.state.board.edges[edge_id].add_rail(1)

        # Player 0 should still be able to place here
        result = setup_manager.validate_rail_placement_forward(
            player_id=0,
            edge_id=edge_id,
        )

        assert result.valid

    def test_place_rail_forward(self, setup_manager: SetupManager):
        """Should place rail and track progress."""
        setup_manager.state.set_phase(Phase.SETUP_RAILS_FORWARD)
        edge_id = (0, 1)
        initial_rails = setup_manager.state.players[0].rail_segments_remaining

        action = setup_manager.place_rail_forward(
            player_id=0,
            edge_id=edge_id,
        )

        # Action should be recorded
        assert action.player_id == 0
        assert action.action_type == "rail"
        assert action.details["phase"] == "forward"

        # Rail should be placed
        assert setup_manager.state.board.edges[edge_id].has_player_rail(0)

        # Player's inventory should decrease
        assert setup_manager.state.players[0].rail_segments_remaining == initial_rails - 1

        # Progress should be tracked
        assert setup_manager._rails_placed_forward[0] is True

    def test_is_rails_forward_phase_complete(self, setup_manager: SetupManager):
        """Should detect when all players completed forward rail placement."""
        setup_manager.state.set_phase(Phase.SETUP_RAILS_FORWARD)
        assert not setup_manager.is_rails_forward_phase_complete()

        # Each player places one rail
        edges = list(setup_manager.state.board.edges.keys())
        for i, player in enumerate(setup_manager.state.players):
            setup_manager.state.global_state.current_player_idx = player.player_id
            setup_manager.place_rail_forward(player.player_id, edges[i])

        assert setup_manager.is_rails_forward_phase_complete()


# =============================================================================
# Reverse Rail Placement Tests
# =============================================================================


class TestReverseRailPlacement:
    """Test reverse rail placement during SETUP_RAILS_REVERSE phase."""

    @pytest.fixture
    def setup_with_forward_rails(self, setup_manager: SetupManager) -> SetupManager:
        """Setup manager with forward rails already placed."""
        setup_manager.state.set_phase(Phase.SETUP_RAILS_FORWARD)

        # Place forward rails for each player on different edges
        edges = list(setup_manager.state.board.edges.keys())
        for i, player in enumerate(setup_manager.state.players):
            setup_manager.state.global_state.current_player_idx = player.player_id
            setup_manager.place_rail_forward(player.player_id, edges[i])

        setup_manager.state.set_phase(Phase.SETUP_RAILS_REVERSE)
        return setup_manager

    def test_get_valid_rail_edges_reverse(self, setup_with_forward_rails: SetupManager):
        """Should return edges connected to player's network endpoints."""
        manager = setup_with_forward_rails

        # Get player 0's first rail edge
        player_edges = manager.state.board.get_player_edges(0)
        assert len(player_edges) == 1

        # Get valid edges for reverse placement
        valid_edges = manager.get_valid_rail_edges_reverse(player_id=0)

        # Should have edges extending from endpoints
        assert len(valid_edges) > 0

        # All valid edges should connect to an endpoint
        endpoints = manager.state.board.get_player_network_endpoints(0)
        for edge_id in valid_edges:
            node_a, node_b = edge_id
            assert node_a in endpoints or node_b in endpoints

    def test_validate_rail_placement_reverse_success(
        self, setup_with_forward_rails: SetupManager
    ):
        """Should validate correct reverse rail placement."""
        manager = setup_with_forward_rails
        manager.state.global_state.current_player_idx = 0

        valid_edges = manager.get_valid_rail_edges_reverse(player_id=0)
        assert len(valid_edges) > 0

        result = manager.validate_rail_placement_reverse(
            player_id=0,
            edge_id=valid_edges[0],
        )

        assert result.valid

    def test_validate_rail_placement_reverse_not_connected(
        self, setup_with_forward_rails: SetupManager
    ):
        """Should reject placement not connected to network."""
        manager = setup_with_forward_rails
        manager.state.global_state.current_player_idx = 0

        # Find an edge not connected to player 0's network
        endpoints = manager.state.board.get_player_network_endpoints(0)
        for edge_id in manager.state.board.edges:
            node_a, node_b = edge_id
            if node_a not in endpoints and node_b not in endpoints:
                result = manager.validate_rail_placement_reverse(
                    player_id=0,
                    edge_id=edge_id,
                )
                assert not result.valid
                assert "not connected" in result.reason.lower()
                break

    def test_place_rail_reverse(self, setup_with_forward_rails: SetupManager):
        """Should place rail and track progress."""
        manager = setup_with_forward_rails
        manager.state.global_state.current_player_idx = 0

        valid_edges = manager.get_valid_rail_edges_reverse(player_id=0)
        edge_id = valid_edges[0]
        initial_rails = manager.state.players[0].rail_segments_remaining

        action = manager.place_rail_reverse(
            player_id=0,
            edge_id=edge_id,
        )

        # Action should be recorded
        assert action.player_id == 0
        assert action.action_type == "rail"
        assert action.details["phase"] == "reverse"

        # Rail should be placed
        assert manager.state.board.edges[edge_id].has_player_rail(0)

        # Player's inventory should decrease
        assert manager.state.players[0].rail_segments_remaining == initial_rails - 1

        # Progress should be tracked
        assert manager._rails_placed_reverse[0] is True

    def test_is_rails_reverse_phase_complete(
        self, setup_with_forward_rails: SetupManager
    ):
        """Should detect when all players completed reverse rail placement."""
        manager = setup_with_forward_rails
        assert not manager.is_rails_reverse_phase_complete()

        # Each player places one reverse rail
        for player in manager.state.players:
            manager.state.global_state.current_player_idx = player.player_id
            valid_edges = manager.get_valid_rail_edges_reverse(player.player_id)
            if valid_edges:
                manager.place_rail_reverse(player.player_id, valid_edges[0])

        assert manager.is_rails_reverse_phase_complete()


# =============================================================================
# Turn Order Tests
# =============================================================================


class TestTurnOrder:
    """Test player turn order helpers."""

    def test_get_buildings_player_order(self, setup_manager: SetupManager):
        """Should return clockwise order from starting player."""
        # Default starting player is 0
        order = setup_manager.get_buildings_player_order()
        assert order == [0, 1, 2]

        # Change starting player
        setup_manager.state.global_state.starting_player_idx = 1
        order = setup_manager.get_buildings_player_order()
        assert order == [1, 2, 0]

    def test_get_rails_forward_player_order(self, setup_manager: SetupManager):
        """Should match building order."""
        buildings_order = setup_manager.get_buildings_player_order()
        forward_order = setup_manager.get_rails_forward_player_order()
        assert forward_order == buildings_order

    def test_get_rails_reverse_player_order(self, setup_manager: SetupManager):
        """Should be reverse of forward order."""
        forward_order = setup_manager.get_rails_forward_player_order()
        reverse_order = setup_manager.get_rails_reverse_player_order()
        assert reverse_order == list(reversed(forward_order))

    def test_get_next_player_buildings(self, setup_manager: SetupManager):
        """Should return next player needing to place buildings."""
        # Initially player 0
        assert setup_manager.get_next_player_buildings() == 0

        # After player 0 places all buildings
        setup_manager._buildings_placed[0] = SetupManager.BUILDINGS_PER_PLAYER
        assert setup_manager.get_next_player_buildings() == 1

        # After all players done
        for p in setup_manager.state.players:
            setup_manager._buildings_placed[p.player_id] = SetupManager.BUILDINGS_PER_PLAYER
        assert setup_manager.get_next_player_buildings() is None

    def test_get_next_player_rails_forward(self, setup_manager: SetupManager):
        """Should return next player needing forward rail placement."""
        assert setup_manager.get_next_player_rails_forward() == 0

        setup_manager._rails_placed_forward[0] = True
        assert setup_manager.get_next_player_rails_forward() == 1

    def test_get_next_player_rails_reverse(self, setup_manager: SetupManager):
        """Should return next player needing reverse rail placement (reverse order)."""
        # Reverse order: 2, 1, 0
        assert setup_manager.get_next_player_rails_reverse() == 2

        setup_manager._rails_placed_reverse[2] = True
        assert setup_manager.get_next_player_rails_reverse() == 1


# =============================================================================
# Setup Summary and Completion Tests
# =============================================================================


class TestSetupCompletion:
    """Test setup completion detection and summary."""

    def test_is_setup_complete(self, setup_manager: SetupManager):
        """Should detect when all setup phases are complete."""
        assert not setup_manager.is_setup_complete()

        # Mark all phases as complete
        for p in setup_manager.state.players:
            setup_manager._buildings_placed[p.player_id] = SetupManager.BUILDINGS_PER_PLAYER
            setup_manager._rails_placed_forward[p.player_id] = True
            setup_manager._rails_placed_reverse[p.player_id] = True

        assert setup_manager.is_setup_complete()

    def test_get_setup_summary(self, setup_manager: SetupManager):
        """Should return correct setup progress summary."""
        summary = setup_manager.get_setup_summary()

        assert "buildings_placed" in summary
        assert "rails_forward_placed" in summary
        assert "rails_reverse_placed" in summary
        assert "buildings_phase_complete" in summary
        assert "setup_complete" in summary
        assert summary["setup_complete"] is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestSetupIntegration:
    """Integration tests for complete setup flow."""

    def test_complete_setup_flow(self, game_state: GameState):
        """Test a complete setup flow with all phases."""
        # Initialize game (places passengers)
        manager = initialize_game(game_state)

        # Verify initial passengers
        parks = game_state.board.get_central_parks()
        total_passengers = sum(len(p.passenger_ids) for p in parks)
        assert total_passengers == len(parks) * INITIAL_PASSENGERS_AT_PARKS

        # Phase 1: Building placement
        assert game_state.phase == Phase.SETUP_BUILDINGS
        for player in game_state.players:
            game_state.global_state.current_player_idx = player.player_id
            for _ in range(SetupManager.BUILDINGS_PER_PLAYER):
                valid_slots = manager.get_valid_building_slots()
                if valid_slots:
                    node_id, slot_idx = valid_slots[0]
                    manager.place_building(
                        player.player_id, node_id, slot_idx, BuildingType.HOUSE
                    )

        assert manager.is_buildings_phase_complete()

        # Transition to forward rails
        game_state.set_phase(Phase.SETUP_RAILS_FORWARD)

        # Phase 2: Forward rail placement
        edges = list(game_state.board.edges.keys())
        for i, player in enumerate(game_state.players):
            game_state.global_state.current_player_idx = player.player_id
            manager.place_rail_forward(player.player_id, edges[i])

        assert manager.is_rails_forward_phase_complete()

        # Transition to reverse rails
        game_state.set_phase(Phase.SETUP_RAILS_REVERSE)

        # Phase 3: Reverse rail placement (reverse order)
        for player_id in manager.get_rails_reverse_player_order():
            game_state.global_state.current_player_idx = player_id
            valid_edges = manager.get_valid_rail_edges_reverse(player_id)
            if valid_edges:
                manager.place_rail_reverse(player_id, valid_edges[0])

        assert manager.is_rails_reverse_phase_complete()
        assert manager.is_setup_complete()

        # Verify final state
        # - Each player should have 2 rail segments placed
        for player in game_state.players:
            assert player.rail_segments_remaining == TOTAL_RAIL_SEGMENTS - 2

        # - Buildings should be placed
        buildings_count = sum(
            1
            for node in game_state.board.nodes.values()
            for slot in node.building_slots
            if slot.building is not None
        )
        expected_buildings = len(game_state.players) * SetupManager.BUILDINGS_PER_PLAYER
        assert buildings_count == expected_buildings

        # - Game state should be valid
        errors = game_state.validate()
        assert len(errors) == 0

    def test_setup_with_shared_forward_rails(self, game_state: GameState):
        """Test that forward rails can be placed on shared edges."""
        manager = initialize_game(game_state)
        game_state.set_phase(Phase.SETUP_RAILS_FORWARD)

        # All players place on the same edge
        edge_id = (0, 1)
        for player in game_state.players:
            game_state.global_state.current_player_idx = player.player_id
            manager.place_rail_forward(player.player_id, edge_id)

        # Edge should have all players' rails
        edge = game_state.board.edges[edge_id]
        assert len(edge.rail_segments) == len(game_state.players)
        for player in game_state.players:
            assert edge.has_player_rail(player.player_id)

    def test_setup_state_validation_after_each_phase(self, game_state: GameState):
        """Game state should remain valid throughout setup."""
        manager = initialize_game(game_state)

        # After passenger placement
        errors = game_state.validate()
        assert len(errors) == 0

        # After some buildings
        valid_slots = manager.get_valid_building_slots()
        manager.place_building(0, valid_slots[0][0], valid_slots[0][1], BuildingType.PUB)
        errors = game_state.validate()
        assert len(errors) == 0

        # After forward rail
        game_state.set_phase(Phase.SETUP_RAILS_FORWARD)
        manager.place_rail_forward(0, (0, 1))
        errors = game_state.validate()
        assert len(errors) == 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
