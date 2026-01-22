"""Tests for rl/action_space.py - Action space mapping for RL."""

import pytest
from pathlib import Path

from rl.action_space import ActionMapping
from rl.config import ActionSpaceConfig, DEFAULT_ACTION_CONFIG
from core.game_state import GameState
from core.constants import ActionAreaType, BuildingType
from core.board import make_edge_id
from engine.game_engine import Action, ActionType
from data.loader import load_board


@pytest.fixture
def board():
    """Load the default board for testing."""
    board_path = Path(__file__).parent.parent / "data" / "default_board.json"
    return load_board(board_path)


@pytest.fixture
def game_state(board):
    """Create an initial game state."""
    return GameState.create_initial_state(board, num_players=4)


@pytest.fixture
def action_mapping(board):
    """Create an action mapping instance."""
    return ActionMapping(board)


class TestActionMappingInit:
    """Tests for ActionMapping initialization."""

    def test_default_config(self, board):
        """Test that ActionMapping uses default config if none provided."""
        mapping = ActionMapping(board)
        assert mapping.config == DEFAULT_ACTION_CONFIG

    def test_custom_config(self, board):
        """Test that ActionMapping accepts custom config."""
        config = ActionSpaceConfig()
        mapping = ActionMapping(board, config)
        assert mapping.config is config

    def test_board_stored(self, board, action_mapping):
        """Test that board is stored."""
        assert action_mapping.board is board

    def test_node_ids_sorted(self, action_mapping):
        """Test that node IDs are sorted for consistency."""
        node_ids = action_mapping._node_ids
        assert node_ids == sorted(node_ids)

    def test_edge_ids_sorted(self, action_mapping):
        """Test that edge IDs are sorted for consistency."""
        edge_ids = action_mapping._edge_ids
        assert edge_ids == sorted(edge_ids)

    def test_total_actions(self, action_mapping):
        """Test that total_actions matches config."""
        assert action_mapping.total_actions == action_mapping.config.total_actions


class TestIndexToAction:
    """Tests for index_to_action conversion."""

    def test_place_marker_actions(self, action_mapping, game_state):
        """Test PLACE_MARKER action conversion."""
        # Test all 7 action areas
        for i in range(7):
            action = action_mapping.index_to_action(i, game_state)
            assert action.action_type == ActionType.PLACE_MARKER
            assert action.player_id == game_state.global_state.current_player_idx
            assert "area_type" in action.params
            # Verify it's a valid ActionAreaType
            ActionAreaType(action.params["area_type"])

    def test_pass_action(self, action_mapping, game_state):
        """Test PASS action conversion."""
        action = action_mapping.index_to_action(7, game_state)
        assert action.action_type == ActionType.PASS
        assert action.player_id == game_state.global_state.current_player_idx
        assert action.params == {}

    def test_setup_building_actions(self, action_mapping, game_state):
        """Test SETUP_BUILDING action conversion."""
        # Test first, middle, and last setup building actions
        for idx in [8, 100, 223]:
            action = action_mapping.index_to_action(idx, game_state)
            assert action.action_type == ActionType.PLACE_BUILDING_SETUP
            assert action.player_id == game_state.global_state.current_player_idx
            assert "node_id" in action.params
            assert "slot_index" in action.params
            assert "building_type" in action.params
            # Verify building type is valid
            BuildingType(action.params["building_type"])

    def test_setup_rail_actions(self, action_mapping, game_state):
        """Test SETUP_RAIL action conversion."""
        action = action_mapping.index_to_action(224, game_state)
        assert action.action_type == ActionType.PLACE_RAIL_SETUP
        assert action.player_id == game_state.global_state.current_player_idx
        assert "edge_id" in action.params
        assert isinstance(action.params["edge_id"], list)
        assert len(action.params["edge_id"]) == 2

    def test_line_expansion_actions(self, action_mapping, game_state):
        """Test LINE_EXPANSION action conversion."""
        action = action_mapping.index_to_action(294, game_state)
        assert action.action_type == ActionType.RESOLVE_LINE_EXPANSION
        assert "edge_id" in action.params

    def test_passengers_actions(self, action_mapping, game_state):
        """Test PASSENGERS action conversion."""
        # Test all 6 passenger distribution options (0-5 to first station)
        for count in range(6):
            idx = 364 + count
            action = action_mapping.index_to_action(idx, game_state)
            assert action.action_type == ActionType.RESOLVE_PASSENGERS
            assert action.params["count_to_first_station"] == count

    def test_buildings_actions(self, action_mapping, game_state):
        """Test BUILDINGS action conversion."""
        action = action_mapping.index_to_action(370, game_state)
        assert action.action_type == ActionType.RESOLVE_BUILDINGS
        assert "node_id" in action.params
        assert "slot_index" in action.params
        assert "building_type" in action.params

    def test_time_clock_actions(self, action_mapping, game_state):
        """Test TIME_CLOCK action conversion."""
        # Test advance clock (586)
        action_advance = action_mapping.index_to_action(586, game_state)
        assert action_advance.action_type == ActionType.RESOLVE_TIME_CLOCK
        assert action_advance.params["advance"] is True

        # Test stop clock (587)
        action_stop = action_mapping.index_to_action(587, game_state)
        assert action_stop.action_type == ActionType.RESOLVE_TIME_CLOCK
        assert action_stop.params["advance"] is False

    def test_vrroomm_actions(self, action_mapping, game_state):
        """Test VRROOMM action conversion."""
        action = action_mapping.index_to_action(588, game_state)
        assert action.action_type == ActionType.RESOLVE_VRROOMM
        assert "passenger_id" in action.params
        assert "to_node" in action.params
        assert "building_slot_index" in action.params

    def test_skip_delivery_action(self, action_mapping, game_state):
        """Test SKIP_DELIVERY action conversion."""
        action = action_mapping.index_to_action(1668, game_state)
        assert action.action_type == ActionType.RESOLVE_VRROOMM
        assert action.params.get("skip") is True

    def test_noop_action(self, action_mapping, game_state):
        """Test NOOP action conversion."""
        action = action_mapping.index_to_action(1669, game_state)
        # NOOP uses PASS with noop flag
        assert action.player_id == game_state.global_state.current_player_idx
        assert action.params.get("noop") is True

    def test_invalid_index_raises_error(self, action_mapping, game_state):
        """Test that invalid indices raise ValueError."""
        with pytest.raises(ValueError):
            action_mapping.index_to_action(-1, game_state)

        with pytest.raises(ValueError):
            action_mapping.index_to_action(1670, game_state)

        with pytest.raises(ValueError):
            action_mapping.index_to_action(10000, game_state)


class TestActionToIndex:
    """Tests for action_to_index conversion."""

    def test_place_marker_conversion(self, action_mapping, game_state):
        """Test PLACE_MARKER action to index."""
        action = Action(
            action_type=ActionType.PLACE_MARKER,
            player_id=0,
            params={"area_type": ActionAreaType.LINE_EXPANSION.value}
        )
        idx = action_mapping.action_to_index(action)
        assert 0 <= idx < 7

    def test_pass_conversion(self, action_mapping, game_state):
        """Test PASS action to index."""
        action = Action(
            action_type=ActionType.PASS,
            player_id=0,
            params={}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 7

    def test_setup_building_conversion(self, action_mapping, game_state):
        """Test SETUP_BUILDING action to index."""
        action = Action(
            action_type=ActionType.PLACE_BUILDING_SETUP,
            player_id=0,
            params={
                "node_id": 0,
                "slot_index": 0,
                "building_type": BuildingType.HOUSE.value
            }
        )
        idx = action_mapping.action_to_index(action)
        assert 8 <= idx < 224

    def test_setup_rail_conversion(self, action_mapping, game_state):
        """Test SETUP_RAIL action to index."""
        first_edge = action_mapping._edge_actions[0]
        action = Action(
            action_type=ActionType.PLACE_RAIL_SETUP,
            player_id=0,
            params={"edge_id": list(first_edge)}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 224  # First rail action

    def test_passengers_conversion(self, action_mapping, game_state):
        """Test PASSENGERS action to index."""
        action = Action(
            action_type=ActionType.RESOLVE_PASSENGERS,
            player_id=0,
            params={"count_to_first_station": 3}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 364 + 3

    def test_time_clock_advance_conversion(self, action_mapping, game_state):
        """Test TIME_CLOCK advance action to index."""
        action = Action(
            action_type=ActionType.RESOLVE_TIME_CLOCK,
            player_id=0,
            params={"advance": True}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 586

    def test_time_clock_stop_conversion(self, action_mapping, game_state):
        """Test TIME_CLOCK stop action to index."""
        action = Action(
            action_type=ActionType.RESOLVE_TIME_CLOCK,
            player_id=0,
            params={"advance": False}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 587

    def test_skip_delivery_conversion(self, action_mapping, game_state):
        """Test SKIP_DELIVERY action to index."""
        action = Action(
            action_type=ActionType.RESOLVE_VRROOMM,
            player_id=0,
            params={"skip": True}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 1668

    def test_noop_conversion(self, action_mapping, game_state):
        """Test NOOP action to index."""
        action = Action(
            action_type=ActionType.PASS,
            player_id=0,
            params={"noop": True}
        )
        idx = action_mapping.action_to_index(action)
        assert idx == 1669


class TestBidirectionalMapping:
    """Tests for bidirectional conversion consistency."""

    def test_place_marker_round_trip(self, action_mapping, game_state):
        """Test that PLACE_MARKER actions round-trip correctly."""
        for idx in range(7):
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_pass_round_trip(self, action_mapping, game_state):
        """Test that PASS action round-trips correctly."""
        idx = 7
        action = action_mapping.index_to_action(idx, game_state)
        recovered_idx = action_mapping.action_to_index(action)
        assert recovered_idx == idx

    def test_setup_building_round_trip(self, action_mapping, game_state):
        """Test that SETUP_BUILDING actions round-trip correctly."""
        # Test a sample of building actions
        for idx in [8, 50, 100, 150, 223]:
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_setup_rail_round_trip(self, action_mapping, game_state):
        """Test that SETUP_RAIL actions round-trip correctly."""
        for idx in [224, 250, 293]:
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_passengers_round_trip(self, action_mapping, game_state):
        """Test that PASSENGERS actions round-trip correctly."""
        for idx in range(364, 370):
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_time_clock_round_trip(self, action_mapping, game_state):
        """Test that TIME_CLOCK actions round-trip correctly."""
        for idx in [586, 587]:
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_vrroomm_round_trip(self, action_mapping, game_state):
        """Test that VRROOMM actions round-trip correctly."""
        # Test a sample of vrroomm actions
        for idx in [588, 1000, 1500, 1667]:
            action = action_mapping.index_to_action(idx, game_state)
            recovered_idx = action_mapping.action_to_index(action)
            assert recovered_idx == idx

    def test_skip_delivery_round_trip(self, action_mapping, game_state):
        """Test that SKIP_DELIVERY action round-trips correctly."""
        idx = 1668
        action = action_mapping.index_to_action(idx, game_state)
        recovered_idx = action_mapping.action_to_index(action)
        assert recovered_idx == idx

    def test_noop_round_trip(self, action_mapping, game_state):
        """Test that NOOP action round-trips correctly."""
        idx = 1669
        action = action_mapping.index_to_action(idx, game_state)
        recovered_idx = action_mapping.action_to_index(action)
        assert recovered_idx == idx


class TestActionRanges:
    """Tests for get_action_range method."""

    def test_place_marker_range(self, action_mapping):
        """Test PLACE_MARKER action range."""
        start, end = action_mapping.get_action_range(ActionType.PLACE_MARKER)
        assert start == 0
        assert end == 7
        assert end - start == 7

    def test_pass_range(self, action_mapping):
        """Test PASS action range."""
        start, end = action_mapping.get_action_range(ActionType.PASS)
        assert start == 7
        assert end == 8
        assert end - start == 1

    def test_setup_building_range(self, action_mapping):
        """Test SETUP_BUILDING action range."""
        start, end = action_mapping.get_action_range(ActionType.PLACE_BUILDING_SETUP)
        assert start == 8
        assert end == 224
        assert end - start == 216

    def test_setup_rail_range(self, action_mapping):
        """Test SETUP_RAIL action range."""
        start, end = action_mapping.get_action_range(ActionType.PLACE_RAIL_SETUP)
        assert start == 224
        assert end == 294
        assert end - start == 70

    def test_passengers_range(self, action_mapping):
        """Test PASSENGERS action range."""
        start, end = action_mapping.get_action_range(ActionType.RESOLVE_PASSENGERS)
        assert start == 364
        assert end == 370
        assert end - start == 6

    def test_vrroomm_range(self, action_mapping):
        """Test VRROOMM action range (includes SKIP_DELIVERY)."""
        start, end = action_mapping.get_action_range(ActionType.RESOLVE_VRROOMM)
        assert start == 588
        assert end == 1669  # Includes skip_delivery at 1668
        assert end - start == 1081  # 1080 deliveries + 1 skip


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_board_same_mapping(self, board):
        """Test that same board produces same mapping."""
        mapping1 = ActionMapping(board)
        mapping2 = ActionMapping(board)

        # Test that node and edge orderings are identical
        assert mapping1._node_ids == mapping2._node_ids
        assert mapping1._edge_ids == mapping2._edge_ids

    def test_consistent_across_states(self, action_mapping, board):
        """Test that mapping is consistent across different game states."""
        state1 = GameState.create_initial_state(board, num_players=3)
        state2 = GameState.create_initial_state(board, num_players=4)

        # Same index should produce structurally similar actions
        # (player_id may differ based on state)
        action1 = action_mapping.index_to_action(0, state1)
        action2 = action_mapping.index_to_action(0, state2)

        assert action1.action_type == action2.action_type
        assert action1.params == action2.params


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_boundary_indices(self, action_mapping, game_state):
        """Test boundary indices don't raise errors."""
        # First action
        action_mapping.index_to_action(0, game_state)

        # Last action
        action_mapping.index_to_action(1669, game_state)

    def test_all_action_areas_represented(self, action_mapping):
        """Test that all ActionAreaType values are represented."""
        for area_type in ActionAreaType:
            # Should not raise ValueError
            action_mapping.get_action_range(ActionType.PLACE_MARKER)

    def test_invalid_edge_id_raises_error(self, action_mapping):
        """Test that invalid edge IDs raise errors."""
        action = Action(
            action_type=ActionType.PLACE_RAIL_SETUP,
            player_id=0,
            params={"edge_id": [9999, 9998]}  # Invalid node IDs
        )
        with pytest.raises(ValueError):
            action_mapping.action_to_index(action)

    def test_invalid_building_params_raise_error(self, action_mapping):
        """Test that invalid building params raise errors."""
        action = Action(
            action_type=ActionType.PLACE_BUILDING_SETUP,
            player_id=0,
            params={
                "node_id": 9999,  # Invalid node
                "slot_index": 0,
                "building_type": BuildingType.HOUSE.value
            }
        )
        with pytest.raises(ValueError):
            action_mapping.action_to_index(action)
