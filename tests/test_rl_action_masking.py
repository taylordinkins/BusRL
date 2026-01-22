"""Tests for rl/action_masking.py - Action mask generation for RL."""

import pytest
import numpy as np
from pathlib import Path

from rl.action_masking import ActionMaskGenerator
from rl.action_space import ActionMapping
from rl.config import ActionSpaceConfig, DEFAULT_ACTION_CONFIG
from core.game_state import GameState
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


@pytest.fixture
def mask_generator(action_mapping):
    """Create an action mask generator."""
    return ActionMaskGenerator(action_mapping)


class TestActionMaskGeneratorInit:
    """Tests for ActionMaskGenerator initialization."""

    def test_default_config(self, action_mapping):
        """Test that mask generator uses default config."""
        generator = ActionMaskGenerator(action_mapping)
        assert generator.config == DEFAULT_ACTION_CONFIG

    def test_custom_config(self, action_mapping):
        """Test that mask generator accepts custom config."""
        config = ActionSpaceConfig()
        generator = ActionMaskGenerator(action_mapping, config)
        assert generator.config is config


class TestGenerateMask:
    """Tests for generate_mask method."""

    def test_returns_numpy_array(self, mask_generator, game_state):
        """Test that generate_mask returns a numpy array."""
        valid_actions = []
        mask = mask_generator.generate_mask(game_state, valid_actions)
        assert isinstance(mask, np.ndarray)

    def test_correct_shape(self, mask_generator, game_state):
        """Test that mask has correct shape."""
        valid_actions = []
        mask = mask_generator.generate_mask(game_state, valid_actions)
        assert mask.shape == (mask_generator.config.total_actions,)

    def test_correct_dtype(self, mask_generator, game_state):
        """Test that mask has boolean dtype."""
        valid_actions = []
        mask = mask_generator.generate_mask(game_state, valid_actions)
        assert mask.dtype == np.bool_

    def test_empty_actions_enables_noop(self, mask_generator, game_state):
        """Test that empty valid actions enables NOOP."""
        mask = mask_generator.generate_mask(game_state, [])
        # NOOP should be enabled when no other actions available
        assert mask[mask_generator.config.noop_idx] == True

    def test_valid_actions_set_to_true(self, mask_generator, game_state):
        """Test that valid actions are marked True in mask."""
        # Create a valid action
        action = Action(
            action_type=ActionType.PASS,
            player_id=0,
            params={}
        )
        mask = mask_generator.generate_mask(game_state, [action])

        # PASS action index should be True
        assert mask[mask_generator.config.pass_idx] == True

    def test_multiple_valid_actions(self, mask_generator, game_state):
        """Test mask with multiple valid actions."""
        actions = [
            Action(ActionType.PASS, player_id=0, params={}),
            Action(ActionType.PLACE_MARKER, player_id=0, params={"area_type": "line_expansion"}),
        ]
        mask = mask_generator.generate_mask(game_state, actions)

        # At least 2 actions should be valid (could be more with NOOP)
        assert np.sum(mask) >= 2


class TestMaskHelpers:
    """Tests for mask helper methods."""

    def test_get_valid_action_indices(self, mask_generator):
        """Test get_valid_action_indices returns correct indices."""
        mask = np.zeros(10, dtype=np.bool_)
        mask[2] = True
        mask[5] = True
        mask[7] = True

        indices = mask_generator.get_valid_action_indices(mask)
        assert list(indices) == [2, 5, 7]

    def test_mask_to_logits_mask(self, mask_generator):
        """Test mask_to_logits_mask conversion."""
        mask = np.array([True, False, True, False], dtype=np.bool_)
        logits_mask = mask_generator.mask_to_logits_mask(mask)

        assert logits_mask.dtype == np.float32
        assert logits_mask[0] == 0.0  # Valid action
        assert logits_mask[1] < -1e7  # Invalid action (large negative)
        assert logits_mask[2] == 0.0  # Valid action
        assert logits_mask[3] < -1e7  # Invalid action

    def test_count_valid_actions(self, mask_generator):
        """Test count_valid_actions returns correct count."""
        mask = np.array([True, False, True, True, False], dtype=np.bool_)
        count = mask_generator.count_valid_actions(mask)
        assert count == 3

    def test_is_action_valid_true(self, mask_generator):
        """Test is_action_valid returns True for valid action."""
        mask = np.array([True, False, True], dtype=np.bool_)
        assert mask_generator.is_action_valid(0, mask) == True
        assert mask_generator.is_action_valid(2, mask) == True

    def test_is_action_valid_false(self, mask_generator):
        """Test is_action_valid returns False for invalid action."""
        mask = np.array([True, False, True], dtype=np.bool_)
        assert mask_generator.is_action_valid(1, mask) == False

    def test_is_action_valid_out_of_range(self, mask_generator):
        """Test is_action_valid returns False for out-of-range index."""
        mask = np.array([True, False, True], dtype=np.bool_)
        assert mask_generator.is_action_valid(-1, mask) == False
        assert mask_generator.is_action_valid(10, mask) == False
