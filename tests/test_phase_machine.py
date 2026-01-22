"""Tests for the phase state machine.

Tests cover:
1. Phase transitions (valid and invalid)
2. Setup phase logic
3. Main game loop transitions
4. End game condition detection
5. Integration with GameState
"""

import pytest

from core.constants import (
    Phase,
    Zone,
    BuildingType,
    ActionAreaType,
    ACTION_RESOLUTION_ORDER,
    TOTAL_ACTION_MARKERS,
)
from core.board import BoardGraph, NodeState, EdgeState, BuildingSlot, make_edge_id
from core.game_state import GameState
from engine.phase_machine import (
    PhaseMachine,
    PhaseTransitionResult,
    PHASE_TRANSITIONS,
)


# =============================================================================
# Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """Test valid and invalid phase transitions."""

    def test_initial_phase(self):
        """Should start in SETUP_BUILDINGS phase."""
        machine = PhaseMachine()
        assert machine.phase == Phase.SETUP_BUILDINGS

    def test_custom_initial_phase(self):
        """Should accept custom initial phase."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        assert machine.phase == Phase.CHOOSING_ACTIONS

    def test_setup_transitions(self):
        """Setup phases should transition linearly."""
        machine = PhaseMachine()

        # SETUP_BUILDINGS -> SETUP_RAILS_FORWARD
        assert machine.can_transition_to(Phase.SETUP_RAILS_FORWARD)
        result = machine.transition_to(Phase.SETUP_RAILS_FORWARD)
        assert result.success
        assert machine.phase == Phase.SETUP_RAILS_FORWARD

        # SETUP_RAILS_FORWARD -> SETUP_RAILS_REVERSE
        assert machine.can_transition_to(Phase.SETUP_RAILS_REVERSE)
        result = machine.transition_to(Phase.SETUP_RAILS_REVERSE)
        assert result.success
        assert machine.phase == Phase.SETUP_RAILS_REVERSE

        # SETUP_RAILS_REVERSE -> CHOOSING_ACTIONS
        assert machine.can_transition_to(Phase.CHOOSING_ACTIONS)
        result = machine.transition_to(Phase.CHOOSING_ACTIONS)
        assert result.success
        assert machine.phase == Phase.CHOOSING_ACTIONS

    def test_main_loop_transitions(self):
        """Main game loop should cycle correctly."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)

        # CHOOSING_ACTIONS -> RESOLVING_ACTIONS
        result = machine.transition_to(Phase.RESOLVING_ACTIONS)
        assert result.success
        assert machine.phase == Phase.RESOLVING_ACTIONS

        # RESOLVING_ACTIONS -> CLEANUP
        result = machine.transition_to(Phase.CLEANUP)
        assert result.success
        assert machine.phase == Phase.CLEANUP

        # CLEANUP -> CHOOSING_ACTIONS (new round)
        result = machine.transition_to(Phase.CHOOSING_ACTIONS)
        assert result.success
        assert machine.phase == Phase.CHOOSING_ACTIONS

    def test_cleanup_to_game_over(self):
        """Cleanup can transition to GAME_OVER."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        assert machine.can_transition_to(Phase.GAME_OVER)
        result = machine.transition_to(Phase.GAME_OVER)
        assert result.success
        assert machine.phase == Phase.GAME_OVER

    def test_game_over_is_terminal(self):
        """GAME_OVER should have no valid transitions."""
        machine = PhaseMachine(initial_phase=Phase.GAME_OVER)

        assert machine.get_valid_transitions() == []
        assert not machine.can_transition_to(Phase.CHOOSING_ACTIONS)

        result = machine.transition_to(Phase.CHOOSING_ACTIONS)
        assert not result.success
        assert result.reason is not None

    def test_invalid_transition_rejected(self):
        """Invalid transitions should be rejected with reason."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)

        # Cannot skip directly to CLEANUP
        assert not machine.can_transition_to(Phase.CLEANUP)
        result = machine.transition_to(Phase.CLEANUP)
        assert not result.success
        assert "Cannot transition" in result.reason

    def test_cannot_go_backwards(self):
        """Cannot transition backwards through phases."""
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)

        assert not machine.can_transition_to(Phase.CHOOSING_ACTIONS)
        result = machine.transition_to(Phase.CHOOSING_ACTIONS)
        assert not result.success

    def test_get_valid_transitions(self):
        """Should return correct valid transitions for each phase."""
        # Check all phases have valid transitions defined
        for phase in Phase:
            machine = PhaseMachine(initial_phase=phase)
            transitions = machine.get_valid_transitions()
            assert transitions == PHASE_TRANSITIONS[phase]


class TestPhaseHelpers:
    """Test phase checking helper methods."""

    def test_is_setup_phase(self):
        """Should correctly identify setup phases."""
        setup_phases = [
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        ]
        non_setup_phases = [
            Phase.CHOOSING_ACTIONS,
            Phase.RESOLVING_ACTIONS,
            Phase.CLEANUP,
            Phase.GAME_OVER,
        ]

        for phase in setup_phases:
            machine = PhaseMachine(initial_phase=phase)
            assert machine.is_setup_phase()

        for phase in non_setup_phases:
            machine = PhaseMachine(initial_phase=phase)
            assert not machine.is_setup_phase()

    def test_is_game_over(self):
        """Should correctly identify game over state."""
        machine = PhaseMachine(initial_phase=Phase.GAME_OVER)
        assert machine.is_game_over()

        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        assert not machine.is_game_over()

    def test_is_choosing_phase(self):
        """Should correctly identify choosing actions phase."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        assert machine.is_choosing_phase()

        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)
        assert not machine.is_choosing_phase()

    def test_is_resolving_phase(self):
        """Should correctly identify resolving actions phase."""
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)
        assert machine.is_resolving_phase()

        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        assert not machine.is_resolving_phase()


# =============================================================================
# Setup Phase Logic Tests
# =============================================================================


class TestSetupPhaseLogic:
    """Test setup phase helper methods."""

    def test_setup_buildings_count(self):
        """Each player should place 2 buildings during setup."""
        machine = PhaseMachine()

        for num_players in [3, 4, 5]:
            counts = machine.get_setup_buildings_count(num_players)
            assert len(counts) == num_players
            for player_id in range(num_players):
                assert counts[player_id] == 2

    def test_setup_rails_forward_order(self):
        """Forward rail placement should be in player order."""
        machine = PhaseMachine()

        assert machine.get_setup_rails_forward_order(3) == [0, 1, 2]
        assert machine.get_setup_rails_forward_order(4) == [0, 1, 2, 3]
        assert machine.get_setup_rails_forward_order(5) == [0, 1, 2, 3, 4]

    def test_setup_rails_reverse_order(self):
        """Reverse rail placement should be in reverse player order."""
        machine = PhaseMachine()

        assert machine.get_setup_rails_reverse_order(3) == [2, 1, 0]
        assert machine.get_setup_rails_reverse_order(4) == [3, 2, 1, 0]
        assert machine.get_setup_rails_reverse_order(5) == [4, 3, 2, 1, 0]


# =============================================================================
# Fixtures for Integration Tests
# =============================================================================


@pytest.fixture
def simple_board() -> BoardGraph:
    """Create a simple test board."""
    graph = BoardGraph()

    # Create 6 nodes: 1 train station, 1 central park, 4 regular
    graph.nodes[0] = NodeState(
        node_id=0,
        is_train_station=True,
        position=(0, 0),
    )
    graph.nodes[1] = NodeState(
        node_id=1,
        is_central_park=True,
        building_slots=[BuildingSlot(zone=Zone.A)],
        position=(1, 0),
    )
    for i in range(2, 6):
        zone = [Zone.A, Zone.B, Zone.C, Zone.D][i - 2]
        graph.nodes[i] = NodeState(
            node_id=i,
            building_slots=[BuildingSlot(zone=zone)],
            position=(i, 0),
        )

    # Connect in a line: 0 -- 1 -- 2 -- 3 -- 4 -- 5
    for i in range(5):
        edge_id = make_edge_id(i, i + 1)
        graph.edges[edge_id] = EdgeState(edge_id=edge_id)
        graph.adjacency.setdefault(i, set()).add(i + 1)
        graph.adjacency.setdefault(i + 1, set()).add(i)

    return graph


@pytest.fixture
def game_state(simple_board: BoardGraph) -> GameState:
    """Create a game state for testing."""
    return GameState.create_initial_state(simple_board, num_players=3)


# =============================================================================
# Integration Tests: Choosing Phase
# =============================================================================


class TestChoosingPhaseTransition:
    """Test choosing actions phase transition logic."""

    def test_should_end_choosing_when_all_passed(self, game_state: GameState):
        """Choosing phase should end when all players have passed."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)

        # Not all passed yet
        assert not machine.should_end_choosing_phase(game_state)

        # All players pass
        for player in game_state.players:
            player.pass_turn()

        assert machine.should_end_choosing_phase(game_state)

    def test_compute_next_phase_choosing_not_ready(self, game_state: GameState):
        """Should not transition if not all players have passed."""
        game_state.set_phase(Phase.CHOOSING_ACTIONS)
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)

        result = machine.compute_next_phase(game_state)
        assert not result.success
        assert "Not all players have passed" in result.reason

    def test_compute_next_phase_choosing_ready(self, game_state: GameState):
        """Should transition to resolving when all have passed."""
        game_state.set_phase(Phase.CHOOSING_ACTIONS)
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)

        for player in game_state.players:
            player.pass_turn()

        result = machine.compute_next_phase(game_state)
        assert result.success
        assert result.new_phase == Phase.RESOLVING_ACTIONS


# =============================================================================
# Integration Tests: Resolving Phase
# =============================================================================


class TestResolvingPhaseTransition:
    """Test resolving actions phase transition logic."""

    def test_should_end_resolving_when_all_areas_done(self, game_state: GameState):
        """Resolving phase should end when all areas are resolved."""
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)

        # Not done yet
        game_state.global_state.current_resolution_area_idx = 0
        assert not machine.should_end_resolving_phase(game_state)

        # All areas resolved
        game_state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)
        assert machine.should_end_resolving_phase(game_state)

    def test_compute_next_phase_resolving_not_ready(self, game_state: GameState):
        """Should not transition if areas remain to resolve."""
        game_state.set_phase(Phase.RESOLVING_ACTIONS)
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)
        game_state.global_state.current_resolution_area_idx = 3

        result = machine.compute_next_phase(game_state)
        assert not result.success
        assert "Not all action areas" in result.reason

    def test_compute_next_phase_resolving_ready(self, game_state: GameState):
        """Should transition to cleanup when all areas resolved."""
        game_state.set_phase(Phase.RESOLVING_ACTIONS)
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)
        game_state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)

        result = machine.compute_next_phase(game_state)
        assert result.success
        assert result.new_phase == Phase.CLEANUP

    def test_get_current_resolution_area(self, game_state: GameState):
        """Should return correct current resolution area."""
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)

        # First area
        game_state.global_state.current_resolution_area_idx = 0
        assert machine.get_current_resolution_area(game_state) == ActionAreaType.LINE_EXPANSION

        # Third area
        game_state.global_state.current_resolution_area_idx = 2
        assert machine.get_current_resolution_area(game_state) == ActionAreaType.PASSENGERS

        # Past end
        game_state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)
        assert machine.get_current_resolution_area(game_state) is None

    def test_get_resolution_progress(self, game_state: GameState):
        """Should return correct resolution progress."""
        machine = PhaseMachine(initial_phase=Phase.RESOLVING_ACTIONS)

        game_state.global_state.current_resolution_area_idx = 3
        current, total = machine.get_resolution_progress(game_state)
        assert current == 3
        assert total == len(ACTION_RESOLUTION_ORDER)


# =============================================================================
# Integration Tests: End Game Detection
# =============================================================================


class TestEndGameDetection:
    """Test end game condition detection."""

    def test_end_game_time_stones_exhausted(self, game_state: GameState):
        """Game should end when all time stones are taken."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # Time stones remaining
        game_state.global_state.time_stones_remaining = 1
        should_end, reason = machine.should_game_end(game_state)
        assert not should_end

        # No time stones remaining
        game_state.global_state.time_stones_remaining = 0
        should_end, reason = machine.should_game_end(game_state)
        assert should_end
        assert "time stones" in reason.lower()

    def test_end_game_only_one_player_with_markers(self, game_state: GameState):
        """Game should end when only one player has markers remaining."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # All players have markers
        should_end, reason = machine.should_game_end(game_state)
        assert not should_end

        # Only one player has markers
        game_state.players[0].action_markers_remaining = 0
        game_state.players[1].action_markers_remaining = 0
        # Player 2 still has markers (default TOTAL_ACTION_MARKERS)

        should_end, reason = machine.should_game_end(game_state)
        assert should_end
        assert "one player has action markers" in reason.lower()

    def test_end_game_all_buildings_placed(self, game_state: GameState):
        """Game should end when all building slots are filled."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # Empty slots exist
        should_end, reason = machine.should_game_end(game_state)
        assert not should_end

        # Fill all slots
        for node in game_state.board.nodes.values():
            for slot in node.building_slots:
                if not slot.building:
                    slot.place_building(BuildingType.HOUSE)

        should_end, reason = machine.should_game_end(game_state)
        assert should_end
        assert "building" in reason.lower()

    def test_compute_next_phase_cleanup_to_game_over(self, game_state: GameState):
        """Should transition to GAME_OVER when end condition met."""
        game_state.set_phase(Phase.CLEANUP)
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # Trigger end condition
        game_state.global_state.time_stones_remaining = 0

        result = machine.compute_next_phase(game_state)
        assert result.success
        assert result.new_phase == Phase.GAME_OVER
        assert result.reason is not None  # Should include end game reason

    def test_compute_next_phase_cleanup_to_new_round(self, game_state: GameState):
        """Should transition to CHOOSING_ACTIONS for new round if game continues."""
        game_state.set_phase(Phase.CLEANUP)
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # No end condition met
        result = machine.compute_next_phase(game_state)
        assert result.success
        assert result.new_phase == Phase.CHOOSING_ACTIONS


# =============================================================================
# Integration Tests: Full Phase Cycle
# =============================================================================


class TestFullPhaseCycle:
    """Integration tests for complete phase cycles."""

    def test_complete_setup_to_game_loop(self, game_state: GameState):
        """Should transition through all setup phases to game loop."""
        machine = PhaseMachine()

        # Verify we start in setup
        assert machine.is_setup_phase()
        assert machine.phase == Phase.SETUP_BUILDINGS

        # Progress through setup
        machine.transition_to(Phase.SETUP_RAILS_FORWARD)
        assert machine.is_setup_phase()

        machine.transition_to(Phase.SETUP_RAILS_REVERSE)
        assert machine.is_setup_phase()

        machine.transition_to(Phase.CHOOSING_ACTIONS)
        assert not machine.is_setup_phase()
        assert machine.is_choosing_phase()

    def test_complete_round_cycle(self, game_state: GameState):
        """Should complete a full round cycle."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        game_state.set_phase(Phase.CHOOSING_ACTIONS)

        # All players pass (end choosing)
        for player in game_state.players:
            player.pass_turn()

        # Transition to resolving
        result = machine.compute_next_phase(game_state)
        assert result.new_phase == Phase.RESOLVING_ACTIONS
        machine.transition_to(Phase.RESOLVING_ACTIONS)

        # Complete all resolution areas
        game_state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)

        # Transition to cleanup
        result = machine.compute_next_phase(game_state)
        assert result.new_phase == Phase.CLEANUP
        machine.transition_to(Phase.CLEANUP)

        # Game continues (no end condition)
        result = machine.compute_next_phase(game_state)
        assert result.new_phase == Phase.CHOOSING_ACTIONS

    def test_multiple_rounds_then_game_over(self, game_state: GameState):
        """Should handle multiple rounds before game ends."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)

        # Simulate several rounds
        for _ in range(3):
            # No end condition - continue to next round
            result = machine.compute_next_phase(game_state)
            assert result.new_phase == Phase.CHOOSING_ACTIONS
            machine.transition_to(Phase.CHOOSING_ACTIONS)
            machine.transition_to(Phase.RESOLVING_ACTIONS)
            machine.transition_to(Phase.CLEANUP)

        # Now trigger end condition
        game_state.global_state.time_stones_remaining = 0

        result = machine.compute_next_phase(game_state)
        assert result.new_phase == Phase.GAME_OVER
        machine.transition_to(Phase.GAME_OVER)

        assert machine.is_game_over()


# =============================================================================
# String Representation Tests
# =============================================================================


class TestStringRepresentation:
    """Test string representations."""

    def test_str(self):
        """Should have readable string representation."""
        machine = PhaseMachine(initial_phase=Phase.CHOOSING_ACTIONS)
        s = str(machine)
        assert "PhaseMachine" in s
        assert "choosing_actions" in s

    def test_repr(self):
        """Should have detailed repr."""
        machine = PhaseMachine(initial_phase=Phase.CLEANUP)
        r = repr(machine)
        assert "PhaseMachine" in r
        assert "Phase.CLEANUP" in r

    def test_transition_result_str(self):
        """PhaseTransitionResult should have useful attributes."""
        result = PhaseTransitionResult(
            success=False,
            new_phase=None,
            reason="Test reason",
        )
        assert not result.success
        assert result.new_phase is None
        assert result.reason == "Test reason"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
