"""Tests for the main game engine.

Tests cover:
1. Game initialization (reset)
2. Action execution (step)
3. Valid action generation (get_valid_actions)
4. Phase transitions through the game flow
5. Integration tests for complete game scenarios
"""

import pytest

from core.constants import (
    Phase,
    ActionAreaType,
    BuildingType,
    Zone,
    MIN_MARKERS_PER_ROUND,
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
    INITIAL_PASSENGERS_AT_PARKS,
)
from core.board import BoardGraph, NodeState, EdgeState, BuildingSlot, make_edge_id
from core.game_state import GameState
from engine.game_engine import (
    GameEngine,
    Action,
    ActionType,
    StepResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_board() -> BoardGraph:
    """Create a simple test board for engine testing."""
    graph = BoardGraph()

    # Train station
    graph.nodes[0] = NodeState(
        node_id=0,
        is_train_station=True,
        position=(0, 0),
    )

    # Central parks (2)
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

    # Regular nodes with Zone A slots (enough for all setup buildings)
    for i in range(3, 10):
        graph.nodes[i] = NodeState(
            node_id=i,
            building_slots=[BuildingSlot(zone=Zone.A)],
            position=(i, 0),
        )

    # Create edges forming a connected graph
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (1, 3), (2, 4), (3, 5), (4, 6),
    ]
    for a, b in edges:
        edge_id = make_edge_id(a, b)
        graph.edges[edge_id] = EdgeState(edge_id=edge_id)
        graph.adjacency.setdefault(a, set()).add(b)
        graph.adjacency.setdefault(b, set()).add(a)

    return graph


@pytest.fixture
def engine(simple_board: BoardGraph) -> GameEngine:
    """Create an initialized game engine."""
    engine = GameEngine()
    engine.reset(num_players=3, board=simple_board)
    return engine


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGameEngineInitialization:
    """Test game engine initialization."""

    def test_reset_creates_state(self, simple_board: BoardGraph):
        """Reset should create a valid initial state."""
        engine = GameEngine()
        state = engine.reset(num_players=3, board=simple_board)

        assert state is not None
        assert len(state.players) == 3
        assert state.phase == Phase.SETUP_BUILDINGS

    def test_reset_places_initial_passengers(self, simple_board: BoardGraph):
        """Reset should place initial passengers at central parks."""
        engine = GameEngine()
        engine.reset(num_players=3, board=simple_board)

        parks = engine.state.board.get_central_parks()
        total_passengers = sum(len(p.passenger_ids) for p in parks)
        expected = len(parks) * INITIAL_PASSENGERS_AT_PARKS
        assert total_passengers == expected

    def test_reset_with_different_player_counts(self, simple_board: BoardGraph):
        """Reset should work with 3, 4, or 5 players."""
        engine = GameEngine()

        for num_players in [3, 4, 5]:
            state = engine.reset(num_players=num_players, board=simple_board)
            assert len(state.players) == num_players

    def test_state_property_raises_before_reset(self):
        """Accessing state before reset should raise error."""
        engine = GameEngine()
        with pytest.raises(RuntimeError):
            _ = engine.state

    def test_is_game_over_false_initially(self, engine: GameEngine):
        """Game should not be over after initialization."""
        assert not engine.is_game_over()


# =============================================================================
# Setup Phase Tests
# =============================================================================


class TestSetupBuildingsPhase:
    """Test building placement during setup."""

    def test_get_valid_actions_returns_building_placements(self, engine: GameEngine):
        """Should return valid building placement actions."""
        actions = engine.get_valid_actions()

        assert len(actions) > 0
        assert all(a.action_type == ActionType.PLACE_BUILDING_SETUP for a in actions)

    def test_valid_actions_include_all_building_types(self, engine: GameEngine):
        """Each valid slot should allow all building types."""
        actions = engine.get_valid_actions()

        # Group by node_id and slot_index
        slot_actions: dict[tuple, list] = {}
        for action in actions:
            key = (action.params["node_id"], action.params["slot_index"])
            slot_actions.setdefault(key, []).append(action)

        # Each slot should have 3 options (one per building type)
        for slot_key, slot_action_list in slot_actions.items():
            assert len(slot_action_list) == len(BuildingType)

    def test_step_places_building(self, engine: GameEngine):
        """Step should place building and update state."""
        actions = engine.get_valid_actions()
        action = actions[0]

        result = engine.step(action)

        assert result.success
        node_id = action.params["node_id"]
        slot_idx = action.params["slot_index"]
        slot = engine.state.board.nodes[node_id].building_slots[slot_idx]
        assert slot.building is not None

    def test_invalid_action_fails(self, engine: GameEngine):
        """Invalid action should fail without changing state."""
        invalid_action = Action(
            action_type=ActionType.PLACE_BUILDING_SETUP,
            player_id=999,  # Invalid player
            params={"node_id": 1, "slot_index": 0, "building_type": "house"},
        )

        result = engine.step(invalid_action)

        assert not result.success
        assert "error" in result.info


class TestSetupRailsPhase:
    """Test rail placement during setup."""

    @pytest.fixture
    def engine_at_rails_forward(self, engine: GameEngine) -> GameEngine:
        """Engine advanced to SETUP_RAILS_FORWARD phase."""
        # Complete building placement for all players
        while engine.state.phase == Phase.SETUP_BUILDINGS:
            actions = engine.get_valid_actions()
            if actions:
                engine.step(actions[0])
            else:
                break

        assert engine.state.phase == Phase.SETUP_RAILS_FORWARD
        return engine

    def test_transition_to_rails_forward(self, engine_at_rails_forward: GameEngine):
        """Should transition to SETUP_RAILS_FORWARD after buildings."""
        assert engine_at_rails_forward.state.phase == Phase.SETUP_RAILS_FORWARD

    def test_get_valid_rail_forward_actions(self, engine_at_rails_forward: GameEngine):
        """Should return valid rail placement actions."""
        actions = engine_at_rails_forward.get_valid_actions()

        assert len(actions) > 0
        assert all(a.action_type == ActionType.PLACE_RAIL_SETUP for a in actions)

    def test_step_places_rail_forward(self, engine_at_rails_forward: GameEngine):
        """Step should place rail and update player inventory."""
        player_id = engine_at_rails_forward.state.global_state.current_player_idx
        initial_rails = engine_at_rails_forward.state.players[player_id].rail_segments_remaining

        actions = engine_at_rails_forward.get_valid_actions()
        result = engine_at_rails_forward.step(actions[0])

        assert result.success
        assert (
            engine_at_rails_forward.state.players[player_id].rail_segments_remaining
            == initial_rails - 1
        )

    @pytest.fixture
    def engine_at_rails_reverse(self, engine_at_rails_forward: GameEngine) -> GameEngine:
        """Engine advanced to SETUP_RAILS_REVERSE phase."""
        # Complete forward rail placement for all players
        while engine_at_rails_forward.state.phase == Phase.SETUP_RAILS_FORWARD:
            actions = engine_at_rails_forward.get_valid_actions()
            if actions:
                engine_at_rails_forward.step(actions[0])
            else:
                break

        assert engine_at_rails_forward.state.phase == Phase.SETUP_RAILS_REVERSE
        return engine_at_rails_forward

    def test_transition_to_rails_reverse(self, engine_at_rails_reverse: GameEngine):
        """Should transition to SETUP_RAILS_REVERSE after forward rails."""
        assert engine_at_rails_reverse.state.phase == Phase.SETUP_RAILS_REVERSE

    def test_get_valid_rail_reverse_actions(self, engine_at_rails_reverse: GameEngine):
        """Should return valid reverse rail placement actions."""
        actions = engine_at_rails_reverse.get_valid_actions()

        assert len(actions) > 0
        assert all(a.action_type == ActionType.PLACE_RAIL_SETUP for a in actions)


# =============================================================================
# Choosing Actions Phase Tests
# =============================================================================


class TestChoosingActionsPhase:
    """Test marker placement during choosing actions phase."""

    @pytest.fixture
    def engine_at_choosing(self, engine: GameEngine) -> GameEngine:
        """Engine advanced to CHOOSING_ACTIONS phase."""
        # Complete all setup phases
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if actions:
                engine.step(actions[0])
            else:
                break

        assert engine.state.phase == Phase.CHOOSING_ACTIONS
        return engine

    def test_transition_to_choosing_actions(self, engine_at_choosing: GameEngine):
        """Should transition to CHOOSING_ACTIONS after setup."""
        assert engine_at_choosing.state.phase == Phase.CHOOSING_ACTIONS

    def test_get_valid_marker_placement_actions(self, engine_at_choosing: GameEngine):
        """Should return valid marker placement actions."""
        actions = engine_at_choosing.get_valid_actions()

        # Should have PLACE_MARKER actions for each available area
        marker_actions = [a for a in actions if a.action_type == ActionType.PLACE_MARKER]
        assert len(marker_actions) > 0

    def test_cannot_pass_before_minimum_markers(self, engine_at_choosing: GameEngine):
        """Should not allow passing before placing MIN_MARKERS_PER_ROUND markers."""
        actions = engine_at_choosing.get_valid_actions()
        pass_actions = [a for a in actions if a.action_type == ActionType.PASS]

        # Player hasn't placed any markers yet
        assert len(pass_actions) == 0

    def test_can_pass_after_minimum_markers(self, engine_at_choosing: GameEngine):
        """Should allow passing after placing MIN_MARKERS_PER_ROUND markers."""
        # Ensure everyone has placed minimum markers so anyone can pass
        num_players = len(engine_at_choosing.state.players)
        total_placements = num_players * MIN_MARKERS_PER_ROUND
        
        for _ in range(total_placements):
            actions = engine_at_choosing.get_valid_actions()
            marker_actions = [a for a in actions if a.action_type == ActionType.PLACE_MARKER]
            if marker_actions:
                engine_at_choosing.step(marker_actions[0])

        # Now should be able to pass
        actions = engine_at_choosing.get_valid_actions()
        pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
        assert len(pass_actions) == 1

    def test_step_places_marker(self, engine_at_choosing: GameEngine):
        """Step should place marker on action board."""
        actions = engine_at_choosing.get_valid_actions()
        marker_action = next(a for a in actions if a.action_type == ActionType.PLACE_MARKER)

        initial_markers = engine_at_choosing.state.action_board.count_total_markers()
        result = engine_at_choosing.step(marker_action)

        assert result.success
        assert (
            engine_at_choosing.state.action_board.count_total_markers()
            == initial_markers + 1
        )

    def test_step_pass_marks_player_passed(self, engine_at_choosing: GameEngine):
        """Step with PASS action should mark player as passed."""
        # Ensure everyone has placed minimum markers so anyone can pass
        num_players = len(engine_at_choosing.state.players)
        total_placements = num_players * MIN_MARKERS_PER_ROUND
        
        for _ in range(total_placements):
            actions = engine_at_choosing.get_valid_actions()
            marker_actions = [a for a in actions if a.action_type == ActionType.PLACE_MARKER]
            if marker_actions:
                engine_at_choosing.step(marker_actions[0])
        
        # Now current player can pass
        player_id = engine_at_choosing.state.global_state.current_player_idx
        
        actions = engine_at_choosing.get_valid_actions()
        pass_action = next(a for a in actions if a.action_type == ActionType.PASS)
        engine_at_choosing.step(pass_action)

        assert engine_at_choosing.state.players[player_id].has_passed

    def test_advances_to_next_player(self, engine_at_choosing: GameEngine):
        """Should advance to next player after action."""
        initial_player = engine_at_choosing.state.global_state.current_player_idx

        actions = engine_at_choosing.get_valid_actions()
        engine_at_choosing.step(actions[0])

        # Should be a different player (unless only one active)
        new_player = engine_at_choosing.state.global_state.current_player_idx
        assert new_player != initial_player or len(engine_at_choosing.state.players) == 1

    def test_no_actions_after_passing(self, engine_at_choosing: GameEngine):
        """Player who passed should have no valid actions."""
        # Place markers and pass for player 0
        while True:
            actions = engine_at_choosing.get_valid_actions()
            if not actions:
                break
            if engine_at_choosing.state.global_state.current_player_idx != 0:
                # Place marker for other players
                marker_actions = [a for a in actions if a.action_type == ActionType.PLACE_MARKER]
                if marker_actions:
                    engine_at_choosing.step(marker_actions[0])
                continue

            # Player 0's turn
            pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
            if pass_actions:
                engine_at_choosing.step(pass_actions[0])
                break
            else:
                engine_at_choosing.step(actions[0])

        # Force back to player 0 and check no actions
        engine_at_choosing.state.global_state.current_player_idx = 0
        assert engine_at_choosing.state.players[0].has_passed
        actions = engine_at_choosing.get_valid_actions()
        assert len(actions) == 0

    def test_can_pass_with_zero_markers_even_if_below_minimum(self, engine_at_choosing: GameEngine):
        """Should allow passing if player has 0 markers, even if they haven't placed minimum."""
        current_player = engine_at_choosing.get_current_player()
        
        # Manually set markers to 0 to simulate running out
        current_player.action_markers_remaining = 0
        current_player.markers_placed_this_round = 0
        
        actions = engine_at_choosing.get_valid_actions()
        
        # Should be able to pass
        pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
        assert len(pass_actions) == 1
        
        # Should NOT be able to place marker
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE_MARKER]
        assert len(place_actions) == 0


# =============================================================================
# Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """Test automatic phase transitions."""

    @pytest.fixture
    def engine_at_choosing(self, engine: GameEngine) -> GameEngine:
        """Engine at CHOOSING_ACTIONS phase."""
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if actions:
                engine.step(actions[0])
        return engine

    def test_transitions_to_resolving_when_all_passed(
        self, engine_at_choosing: GameEngine
    ):
        """Should transition to RESOLVING_ACTIONS when all players pass."""
        # All players place minimum markers then pass
        while engine_at_choosing.state.phase == Phase.CHOOSING_ACTIONS:
            actions = engine_at_choosing.get_valid_actions()
            if not actions:
                break

            # Prefer pass if available, otherwise place marker
            pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
            if pass_actions:
                engine_at_choosing.step(pass_actions[0])
            else:
                engine_at_choosing.step(actions[0])

        assert engine_at_choosing.state.phase == Phase.RESOLVING_ACTIONS


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_current_player(self, engine: GameEngine):
        """Should return current player."""
        player = engine.get_current_player()
        assert player.player_id == engine.state.global_state.current_player_idx

    def test_get_max_buses(self, engine: GameEngine):
        """Should return maximum buses among all players."""
        # Initially all players have 1 bus
        assert engine.get_max_buses() == 1

        # Give one player more buses
        engine.state.players[0].gain_bus()
        engine.state.players[0].gain_bus()
        assert engine.get_max_buses() == 3

    def test_clone_creates_independent_copy(self, engine: GameEngine):
        """Clone should create independent engine copy."""
        cloned = engine.clone()

        # Modify original
        actions = engine.get_valid_actions()
        engine.step(actions[0])

        # Clone should be unaffected
        assert cloned.state.phase == Phase.SETUP_BUILDINGS
        # Original advanced (may have placed a building)

    def test_get_game_summary(self, engine: GameEngine):
        """Should return comprehensive game summary."""
        summary = engine.get_game_summary()

        assert "phase" in summary
        assert "round" in summary
        assert "current_player" in summary
        assert "time_clock" in summary
        assert "players" in summary
        assert "game_over" in summary
        assert len(summary["players"]) == 3

    def test_str_representation(self, engine: GameEngine):
        """Should have readable string representation."""
        s = str(engine)
        assert "GameEngine" in s
        assert "setup_buildings" in s

    def test_str_before_reset(self):
        """String representation should work before reset."""
        engine = GameEngine()
        s = str(engine)
        assert "not initialized" in s


# =============================================================================
# Integration Tests
# =============================================================================


class TestGameEngineIntegration:
    """Integration tests for complete game scenarios."""

    def test_complete_setup_phases(self, engine: GameEngine):
        """Should complete all setup phases."""
        phases_seen = [engine.state.phase]

        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if not actions:
                break
            result = engine.step(actions[0])
            assert result.success

            if engine.state.phase not in phases_seen:
                phases_seen.append(engine.state.phase)

        # Should have seen all setup phases and arrived at CHOOSING_ACTIONS
        assert Phase.SETUP_BUILDINGS in phases_seen
        assert Phase.SETUP_RAILS_FORWARD in phases_seen
        assert Phase.SETUP_RAILS_REVERSE in phases_seen
        assert engine.state.phase == Phase.CHOOSING_ACTIONS

    def test_state_remains_valid_throughout_setup(self, engine: GameEngine):
        """State should remain valid through all setup actions."""
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            # Validate state before action
            errors = engine.state.validate()
            assert len(errors) == 0, f"State invalid: {errors}"

            actions = engine.get_valid_actions()
            if not actions:
                break
            engine.step(actions[0])

        # Final validation
        errors = engine.state.validate()
        assert len(errors) == 0

    def test_players_have_correct_resources_after_setup(self, engine: GameEngine):
        """Players should have correct resources after setup."""
        # Complete setup
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if not actions:
                break
            engine.step(actions[0])

        # Each player should have placed 2 rails
        for player in engine.state.players:
            assert player.rail_segments_remaining == TOTAL_RAIL_SEGMENTS - 2

        # All players should have full action markers
        for player in engine.state.players:
            assert player.action_markers_remaining == TOTAL_ACTION_MARKERS

    def test_complete_choosing_to_resolving_transition(self, engine: GameEngine):
        """Should complete a full choosing phase and transition to resolving."""
        # Complete setup first
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if not actions:
                break
            engine.step(actions[0])

        assert engine.state.phase == Phase.CHOOSING_ACTIONS

        # Complete choosing phase
        while engine.state.phase == Phase.CHOOSING_ACTIONS:
            actions = engine.get_valid_actions()
            if not actions:
                break

            # Prefer pass if available
            pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
            if pass_actions:
                engine.step(pass_actions[0])
            else:
                engine.step(actions[0])

        assert engine.state.phase == Phase.RESOLVING_ACTIONS

    def test_action_board_has_markers_after_choosing(self, engine: GameEngine):
        """Action board should have markers after choosing phase."""
        # Complete setup
        while engine.state.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ):
            actions = engine.get_valid_actions()
            if not actions:
                break
            engine.step(actions[0])

        # Complete choosing (each player places at least 2 markers)
        while engine.state.phase == Phase.CHOOSING_ACTIONS:
            actions = engine.get_valid_actions()
            if not actions:
                break

            pass_actions = [a for a in actions if a.action_type == ActionType.PASS]
            if pass_actions:
                engine.step(pass_actions[0])
            else:
                engine.step(actions[0])

        # Should have at least 2 * num_players markers
        total_markers = engine.state.action_board.count_total_markers()
        min_expected = MIN_MARKERS_PER_ROUND * len(engine.state.players)
        assert total_markers >= min_expected


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
