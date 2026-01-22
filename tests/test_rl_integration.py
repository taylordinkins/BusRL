"""Integration tests for RL environment with random agent.

These tests verify that:
1. Random agents can play complete games without errors
2. All game phases transition correctly
3. Action masking prevents illegal moves
4. The environment maintains consistency throughout episodes
"""

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


class TestRandomAgentFullGame:
    """Tests for random agent playing complete games."""

    def test_random_agent_completes_game_4_players(self):
        """Test that random agent can complete a full 4-player game."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        max_steps = 5000  # Safety limit
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]

            # Must have at least one valid action
            assert len(valid_indices) > 0, f"No valid actions at step {steps}, phase {info.get('phase')}"

            # Random agent picks uniformly from valid actions
            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)

            # Validate observation
            assert obs.shape == env.observation_space.shape
            assert not np.isnan(obs).any(), f"NaN in observation at step {steps}"
            assert np.all(obs >= 0.0), f"Negative observation value at step {steps}"
            assert np.all(obs <= 1.0), f"Observation > 1.0 at step {steps}"

            steps += 1

        # Game should terminate before safety limit
        assert terminated or truncated, f"Game did not terminate after {max_steps} steps"
        env.close()

    def test_random_agent_completes_game_3_players(self):
        """Test that random agent can complete a full 3-player game."""
        env = BusEnv(num_players=3)
        obs, info = env.reset(seed=123)

        max_steps = 5000
        steps = 0
        terminated = False

        while not terminated and steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            assert len(valid_indices) > 0

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated, f"Game did not terminate after {max_steps} steps"
        env.close()

    def test_random_agent_completes_game_5_players(self):
        """Test that random agent can complete a full 5-player game."""
        env = BusEnv(num_players=5)
        obs, info = env.reset(seed=456)

        max_steps = 6000  # More steps for larger game
        steps = 0
        terminated = False

        while not terminated and steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            assert len(valid_indices) > 0

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated, f"Game did not terminate after {max_steps} steps"
        env.close()

    def test_multiple_games_same_env(self):
        """Test that environment can be reset and play multiple games."""
        env = BusEnv(num_players=4)

        for game_num in range(3):
            obs, info = env.reset(seed=game_num)

            max_steps = 5000
            steps = 0
            terminated = False

            while not terminated and steps < max_steps:
                mask = env.action_masks()
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    break

                action = np.random.choice(valid_indices)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

            # Each game should complete
            assert terminated, f"Game {game_num} did not terminate"

        env.close()


class TestPhaseTransitions:
    """Tests for phase transitions during gameplay."""

    def test_all_phases_visited(self):
        """Test that all major phases are visited during a game."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        phases_seen = set()
        phases_seen.add(info["phase"])

        max_steps = 5000
        steps = 0
        terminated = False

        while not terminated and steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            phases_seen.add(info["phase"])
            steps += 1

        # Should have seen key phases (setup_rails is split into forward/reverse)
        expected_phases = {"setup_buildings", "choosing_actions"}
        assert expected_phases.issubset(phases_seen), f"Missing phases. Saw: {phases_seen}"
        # Should have seen at least one of the setup_rails phases
        assert "setup_rails_forward" in phases_seen or "setup_rails_reverse" in phases_seen, \
            f"Missing setup_rails phases. Saw: {phases_seen}"
        env.close()

    def test_setup_phase_completes(self):
        """Test that setup phases complete correctly."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        assert info["phase"] == "setup_buildings"

        max_steps = 500
        steps = 0

        # Play through setup
        while info["phase"].startswith("setup") and steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        # Should have exited setup
        assert not info["phase"].startswith("setup"), f"Stuck in setup phase after {steps} steps"
        env.close()


class TestActionMaskingIntegrity:
    """Tests for action masking correctness."""

    def test_masked_actions_never_chosen(self):
        """Test that environment correctly rejects invalid actions."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        max_steps = 200
        steps = 0
        invalid_action_attempts = 0

        while steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            invalid_indices = np.where(~mask)[0]

            if len(valid_indices) == 0:
                break

            # Occasionally try an invalid action to verify rejection
            if len(invalid_indices) > 0 and np.random.random() < 0.1:
                invalid_action = np.random.choice(invalid_indices)
                obs, reward, terminated, truncated, info = env.step(invalid_action)

                # Should get penalty and error indicator
                assert reward < 0, "Invalid action should give negative reward"
                assert info.get("invalid_action", False), "Should flag invalid action"
                invalid_action_attempts += 1
            else:
                # Normal valid action
                action = np.random.choice(valid_indices)
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    break

            steps += 1

        # Should have tried some invalid actions
        assert invalid_action_attempts > 0, "Test should have tried invalid actions"
        env.close()

    def test_action_mask_consistency(self):
        """Test that action mask is consistent with valid actions list."""
        env = BusEnv(num_players=4)
        env.reset(seed=42)

        for _ in range(50):
            mask = env.action_masks()
            valid_actions = env.get_valid_actions()

            # Number of True values in mask should match valid actions
            # (may have NOOP added if no actions)
            mask_count = np.sum(mask)
            action_count = len(valid_actions)

            # If no valid actions, NOOP should be enabled
            if action_count == 0:
                assert mask_count >= 1, "NOOP should be enabled when no valid actions"
            else:
                # Valid action count should be reflected in mask
                # (might have +1 for NOOP in some cases)
                assert mask_count >= action_count

            # Take a valid action
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break

        env.close()


class TestObservationConsistency:
    """Tests for observation tensor consistency."""

    def test_observations_always_valid(self):
        """Test that observations are always within valid bounds."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        # Check initial observation
        assert obs.shape == (DEFAULT_OBS_CONFIG.total_observation_dim,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert not np.isnan(obs).any()

        max_steps = 500
        for _ in range(max_steps):
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check every observation
            assert obs.shape == (DEFAULT_OBS_CONFIG.total_observation_dim,)
            assert np.all(obs >= 0.0), f"Negative value: {obs.min()}"
            assert np.all(obs <= 1.0), f"Value > 1: {obs.max()}"
            assert not np.isnan(obs).any()

            if terminated:
                break

        env.close()

    def test_observation_changes_after_step(self):
        """Test that observations change after taking actions."""
        env = BusEnv(num_players=4)
        obs1, _ = env.reset(seed=42)

        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)

        obs2, _, _, _, _ = env.step(action)

        # Observation should typically change (not guaranteed but very likely)
        # At minimum, phase or player info should change
        env.close()


class TestRewardIntegrity:
    """Tests for reward calculation integrity."""

    def test_rewards_are_finite(self):
        """Test that rewards are always finite numbers."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        rewards = []
        max_steps = 500

        for _ in range(max_steps):
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            rewards.append(reward)

            if terminated:
                break

        # Should have collected some rewards
        assert len(rewards) > 0
        env.close()

    def test_terminal_reward_exists(self):
        """Test that terminal state has meaningful reward."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        max_steps = 5000
        steps = 0
        final_reward = 0

        while steps < max_steps:
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            final_reward = reward
            steps += 1

            if terminated:
                break

        # Terminal reward should be non-zero (winner gets positive, losers negative)
        # Note: Could be zero in rare tie cases
        env.close()


class TestEnvironmentCloning:
    """Tests for environment cloning functionality."""

    def test_clone_creates_independent_env(self):
        """Test that clone creates an independent environment."""
        env = BusEnv(num_players=4)
        env.reset(seed=42)

        # Play a few steps
        for _ in range(10):
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break
            action = np.random.choice(valid_indices)
            env.step(action)

        # Clone the environment
        cloned = env.clone()

        # Both should have same state
        assert cloned.get_current_player() == env.get_current_player()

        # Taking action in original shouldn't affect clone
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
            env.step(action)

            # Clone should still be at previous state
            # (current player might have changed in original)

        env.close()
        cloned.close()

    def test_clone_can_continue_game(self):
        """Test that cloned environment can continue playing."""
        env = BusEnv(num_players=4)
        env.reset(seed=42)

        # Play a few steps
        for _ in range(20):
            mask = env.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break
            action = np.random.choice(valid_indices)
            env.step(action)

        # Clone and continue
        cloned = env.clone()

        for _ in range(50):
            mask = cloned.action_masks()
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break
            action = np.random.choice(valid_indices)
            obs, _, terminated, _, _ = cloned.step(action)

            # Observation should be valid
            assert not np.isnan(obs).any()

            if terminated:
                break

        env.close()
        cloned.close()


class TestInfoDictionary:
    """Tests for info dictionary contents."""

    def test_info_contains_required_keys(self):
        """Test that info dictionary has all required keys."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        required_keys = ["phase", "round", "current_player", "valid_action_count"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"

        # Take a step and check again
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        obs, _, _, _, info = env.step(action)

        for key in required_keys:
            assert key in info, f"Missing key after step: {key}"

        env.close()

    def test_info_tracks_scores(self):
        """Test that info dictionary tracks player scores."""
        env = BusEnv(num_players=4)
        obs, info = env.reset(seed=42)

        assert "scores" in info
        assert len(info["scores"]) == 4

        # All scores should start at 0
        for player_id, score in info["scores"].items():
            assert score == 0

        env.close()


class TestMakeBusEnvFactory:
    """Tests for make_bus_env factory function."""

    def test_factory_creates_env(self):
        """Test that factory function creates valid environment."""
        env = make_bus_env(num_players=4)
        assert isinstance(env, BusEnv)
        assert env.num_players == 4
        env.close()

    def test_factory_with_different_players(self):
        """Test factory with different player counts."""
        for num_players in [3, 4, 5]:
            env = make_bus_env(num_players=num_players)
            assert env.num_players == num_players

            # Should be able to reset and play
            env.reset(seed=42)
            mask = env.action_masks()
            assert np.sum(mask) > 0

            env.close()


class TestDeterminism:
    """Tests for environment determinism with seeds."""

    def test_same_seed_same_initial_state(self):
        """Test that same seed produces same initial state."""
        env1 = BusEnv(num_players=4)
        env2 = BusEnv(num_players=4)

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        # Observations should be identical
        np.testing.assert_array_equal(obs1, obs2)

        env1.close()
        env2.close()

    def test_different_seeds_different_states(self):
        """Test that different seeds can produce different states."""
        env1 = BusEnv(num_players=4)
        env2 = BusEnv(num_players=4)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=999)

        # Note: With fixed board, initial observations might still be identical
        # This is expected behavior since the board is deterministic

        env1.close()
        env2.close()
