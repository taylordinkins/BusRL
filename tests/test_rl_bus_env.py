"""Tests for rl/bus_env.py - Gymnasium environment for Bus."""

import pytest
import numpy as np
from pathlib import Path

from rl.bus_env import BusEnv, make_bus_env
from rl.config import DEFAULT_OBS_CONFIG, DEFAULT_ACTION_CONFIG
from core.constants import Phase
from data.loader import load_board


@pytest.fixture
def board():
    """Load the default board for testing."""
    board_path = Path(__file__).parent.parent / "data" / "default_board.json"
    return load_board(board_path)


@pytest.fixture
def env():
    """Create a BusEnv instance."""
    return BusEnv(num_players=4)


@pytest.fixture
def reset_env(env):
    """Create and reset a BusEnv instance."""
    env.reset()
    return env


class TestBusEnvInit:
    """Tests for BusEnv initialization."""

    def test_default_num_players(self):
        """Test default number of players."""
        env = BusEnv()
        assert env.num_players == 4

    def test_custom_num_players(self):
        """Test custom number of players."""
        env = BusEnv(num_players=3)
        assert env.num_players == 3

    def test_observation_space_shape(self, env):
        """Test observation space has correct shape."""
        expected_shape = (DEFAULT_OBS_CONFIG.total_observation_dim,)
        assert env.observation_space.shape == expected_shape

    def test_observation_space_bounds(self, env):
        """Test observation space bounds are [0, 1]."""
        assert env.observation_space.low[0] == 0.0
        assert env.observation_space.high[0] == 1.0

    def test_action_space_size(self, env):
        """Test action space has correct size."""
        assert env.action_space.n == DEFAULT_ACTION_CONFIG.total_actions

    def test_make_bus_env_factory(self):
        """Test make_bus_env factory function."""
        env = make_bus_env(num_players=5)
        assert isinstance(env, BusEnv)
        assert env.num_players == 5


class TestBusEnvReset:
    """Tests for reset method."""

    def test_reset_returns_tuple(self, env):
        """Test that reset returns (observation, info) tuple."""
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_returns_observation(self, env):
        """Test that reset returns valid observation."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_reset_returns_info_dict(self, env):
        """Test that reset returns info dictionary."""
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert "phase" in info
        assert "current_player" in info

    def test_reset_initializes_game(self, env):
        """Test that reset initializes the game state."""
        env.reset()
        state = env.get_state()
        assert state is not None
        assert state.phase == Phase.SETUP_BUILDINGS

    def test_reset_with_seed(self, env):
        """Test that reset accepts seed parameter."""
        obs1, _ = env.reset(seed=42)
        # Reset should work with seed (deterministic initialization)
        obs2, _ = env.reset(seed=42)
        # Note: Observations should be similar but board is always the same
        assert obs1.shape == obs2.shape


class TestBusEnvStep:
    """Tests for step method."""

    def test_step_returns_tuple(self, reset_env):
        """Test that step returns 5-tuple."""
        mask = reset_env.action_masks()
        valid_idx = np.where(mask)[0][0]
        result = reset_env.step(valid_idx)

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_returns_correct_types(self, reset_env):
        """Test that step returns correct types."""
        mask = reset_env.action_masks()
        valid_idx = np.where(mask)[0][0]
        obs, reward, terminated, truncated, info = reset_env.step(valid_idx)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self, reset_env):
        """Test that step returns observation with correct shape."""
        mask = reset_env.action_masks()
        valid_idx = np.where(mask)[0][0]
        obs, _, _, _, _ = reset_env.step(valid_idx)

        assert obs.shape == reset_env.observation_space.shape

    def test_step_info_contains_metadata(self, reset_env):
        """Test that step info contains game metadata."""
        mask = reset_env.action_masks()
        valid_idx = np.where(mask)[0][0]
        _, _, _, _, info = reset_env.step(valid_idx)

        assert "phase" in info
        assert "round" in info
        assert "current_player" in info
        assert "valid_action_count" in info


class TestBusEnvActionMasks:
    """Tests for action_masks method."""

    def test_action_masks_returns_array(self, reset_env):
        """Test that action_masks returns numpy array."""
        mask = reset_env.action_masks()
        assert isinstance(mask, np.ndarray)

    def test_action_masks_correct_shape(self, reset_env):
        """Test that action_masks has correct shape."""
        mask = reset_env.action_masks()
        assert mask.shape == (DEFAULT_ACTION_CONFIG.total_actions,)

    def test_action_masks_boolean_dtype(self, reset_env):
        """Test that action_masks has boolean dtype."""
        mask = reset_env.action_masks()
        assert mask.dtype == np.bool_

    def test_action_masks_has_valid_actions(self, reset_env):
        """Test that action_masks has at least one valid action."""
        mask = reset_env.action_masks()
        assert np.sum(mask) > 0

    def test_action_masks_before_reset(self, env):
        """Test action_masks returns all-invalid before reset."""
        mask = env.action_masks()
        assert np.sum(mask) == 0


class TestBusEnvRender:
    """Tests for render method."""

    def test_render_ansi_returns_string(self, reset_env):
        """Test that render with ansi mode returns string."""
        reset_env.render_mode = "ansi"
        result = reset_env.render()
        assert isinstance(result, str)

    def test_render_none_returns_none(self, reset_env):
        """Test that render with no mode returns None."""
        reset_env.render_mode = None
        result = reset_env.render()
        assert result is None


class TestBusEnvUtilities:
    """Tests for utility methods."""

    def test_get_state(self, reset_env):
        """Test get_state returns game state."""
        state = reset_env.get_state()
        assert state is not None
        assert hasattr(state, "phase")
        assert hasattr(state, "players")

    def test_get_valid_actions(self, reset_env):
        """Test get_valid_actions returns list."""
        actions = reset_env.get_valid_actions()
        assert isinstance(actions, list)
        # During setup, there should be valid actions
        assert len(actions) > 0

    def test_get_current_player(self, reset_env):
        """Test get_current_player returns player id."""
        player_id = reset_env.get_current_player()
        assert isinstance(player_id, int)
        assert 0 <= player_id < reset_env.num_players

    def test_clone_creates_copy(self, reset_env):
        """Test clone creates independent copy."""
        cloned = reset_env.clone()

        # Should be independent instances
        assert cloned is not reset_env
        assert cloned._engine is not reset_env._engine

        # But same game state content
        assert cloned.get_current_player() == reset_env.get_current_player()

    def test_close(self, reset_env):
        """Test close cleans up resources."""
        reset_env.close()
        assert reset_env._engine is None


class TestBusEnvIntegration:
    """Integration tests for full episode flow."""

    def test_can_take_multiple_steps(self, reset_env):
        """Test that environment can handle multiple steps."""
        for _ in range(10):
            mask = reset_env.action_masks()
            if np.sum(mask) == 0:
                break
            valid_idx = np.where(mask)[0][0]
            obs, reward, terminated, truncated, info = reset_env.step(valid_idx)

            if terminated or truncated:
                break

            # Observation should always be valid
            assert obs.shape == reset_env.observation_space.shape
            assert not np.isnan(obs).any()

    def test_setup_phase_actions(self, reset_env):
        """Test that setup phase has building placement actions."""
        info = reset_env._get_info()
        assert info["phase"] == "setup_buildings"

        # Should have valid building setup actions
        valid_actions = reset_env.get_valid_actions()
        assert len(valid_actions) > 0

    def test_game_starts_not_terminated(self, reset_env):
        """Test that game doesn't start terminated."""
        mask = reset_env.action_masks()
        valid_idx = np.where(mask)[0][0]
        _, _, terminated, truncated, _ = reset_env.step(valid_idx)

        # Game shouldn't be over after one action
        assert not terminated
        assert not truncated

    def test_observations_in_valid_range(self, reset_env):
        """Test that all observations are in [0, 1] range."""
        for _ in range(5):
            mask = reset_env.action_masks()
            if np.sum(mask) == 0:
                break
            valid_idx = np.where(mask)[0][0]
            obs, _, terminated, _, _ = reset_env.step(valid_idx)

            assert np.all(obs >= 0.0), f"Min obs value: {obs.min()}"
            assert np.all(obs <= 1.0), f"Max obs value: {obs.max()}"

            if terminated:
                break
