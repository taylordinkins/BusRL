"""Tests for rl/reward.py - Reward calculation for RL."""

import pytest
from pathlib import Path

from rl.reward import RewardCalculator, StepRewardInfo
from rl.config import RewardConfig, BoardConfig, DEFAULT_REWARD_CONFIG
from core.game_state import GameState
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
def reward_calculator():
    """Create a reward calculator with default config."""
    return RewardCalculator()


class TestRewardCalculatorInit:
    """Tests for RewardCalculator initialization."""

    def test_default_config(self):
        """Test that calculator uses default config."""
        calc = RewardCalculator()
        assert calc.config == DEFAULT_REWARD_CONFIG

    def test_custom_config(self):
        """Test that calculator accepts custom config."""
        config = RewardConfig(delivery_reward=2.0)
        calc = RewardCalculator(config=config)
        assert calc.config.delivery_reward == 2.0

    def test_reset_clears_state(self):
        """Test that reset clears tracking state."""
        calc = RewardCalculator()
        calc._stations_connected[0] = {8, 27}
        calc.reset()
        assert calc._stations_connected == {}


class TestStepRewardInfo:
    """Tests for StepRewardInfo dataclass."""

    def test_total_calculation(self):
        """Test that total sums all reward components."""
        info = StepRewardInfo(
            base_reward=1.0,
            delivery_reward=2.0,
            stolen_bonus=0.5,
            exclusive_bonus=0.1,
            station_connection_reward=0.2,
            time_stone_penalty=-0.5,
            terminal_reward=3.0,
        )
        expected = 1.0 + 2.0 + 0.5 + 0.1 + 0.2 - 0.5 + 3.0
        assert info.total == expected

    def test_default_values_zero(self):
        """Test that default values are all zero."""
        info = StepRewardInfo()
        assert info.total == 0.0


class TestDeliveryReward:
    """Tests for delivery reward calculation."""

    def test_no_score_change_no_reward(self, reward_calculator, game_state):
        """Test that no score change gives no delivery reward."""
        prev_state = game_state.clone()
        # No changes to state
        reward = reward_calculator._compute_delivery_reward(game_state, prev_state, 0)
        assert reward == 0.0

    def test_score_increase_gives_reward(self, reward_calculator, game_state):
        """Test that score increase gives delivery reward."""
        prev_state = game_state.clone()
        # Manually increase player 0's score
        game_state.players[0].score = 3

        reward = reward_calculator._compute_delivery_reward(game_state, prev_state, 0)
        # 3 points * 1.0 reward per point
        assert reward == 3.0

    def test_reward_scales_with_config(self, game_state):
        """Test that reward scales with config value."""
        config = RewardConfig(delivery_reward=0.5)
        calc = RewardCalculator(config=config)

        prev_state = game_state.clone()
        game_state.players[0].score = 2

        reward = calc._compute_delivery_reward(game_state, prev_state, 0)
        assert reward == 1.0  # 2 points * 0.5


class TestTimeStoneReward:
    """Tests for time stone penalty calculation."""

    def test_no_time_stone_no_penalty(self, reward_calculator, game_state):
        """Test that no time stone change gives no penalty."""
        prev_state = game_state.clone()
        penalty = reward_calculator._compute_time_stone_penalty(game_state, prev_state, 0)
        assert penalty == 0.0

    def test_time_stone_gives_penalty(self, reward_calculator, game_state):
        """Test that taking a time stone gives penalty."""
        prev_state = game_state.clone()
        game_state.players[0].time_stones = 1

        penalty = reward_calculator._compute_time_stone_penalty(game_state, prev_state, 0)
        assert penalty == -0.01  # Default penalty

    def test_penalty_scales_with_config(self, game_state):
        """Test that penalty scales with config value."""
        config = RewardConfig(time_stone_penalty=-0.1)
        calc = RewardCalculator(config=config)

        prev_state = game_state.clone()
        game_state.players[0].time_stones = 2

        penalty = calc._compute_time_stone_penalty(game_state, prev_state, 0)
        assert penalty == -0.2  # 2 stones * -0.1


class TestTerminalReward:
    """Tests for terminal reward calculation."""

    def test_winner_gets_positive_reward(self, reward_calculator, game_state):
        """Test that winner gets positive reward."""
        # Set up scores: player 0 wins with 10 points
        game_state.players[0].score = 10
        game_state.players[1].score = 5
        game_state.players[2].score = 3
        game_state.players[3].score = 1

        reward = reward_calculator._compute_terminal_reward(game_state, 0)
        # Winner: 10 - 5 (second place) = 5
        assert reward == 5.0

    def test_losers_get_negative_reward(self, reward_calculator, game_state):
        """Test that losers get negative reward."""
        game_state.players[0].score = 10
        game_state.players[1].score = 5
        game_state.players[2].score = 3
        game_state.players[3].score = 1

        # Player 1's reward: 5 - 10 = -5
        reward1 = reward_calculator._compute_terminal_reward(game_state, 1)
        assert reward1 == -5.0

        # Player 3's reward: 1 - 10 = -9
        reward3 = reward_calculator._compute_terminal_reward(game_state, 3)
        assert reward3 == -9.0

    def test_time_stones_subtracted_from_final_score(self, reward_calculator, game_state):
        """Test that time stones are subtracted from final score."""
        game_state.players[0].score = 10
        game_state.players[0].time_stones = 3  # Final: 10 - 3 = 7
        game_state.players[1].score = 8
        game_state.players[1].time_stones = 0  # Final: 8

        # Player 1 should win: 8 > 7
        reward1 = reward_calculator._compute_terminal_reward(game_state, 1)
        assert reward1 == 1.0  # 8 - 7 = 1

        # Player 0 should lose
        reward0 = reward_calculator._compute_terminal_reward(game_state, 0)
        assert reward0 == -1.0  # 7 - 8 = -1


class TestStationConnectionReward:
    """Tests for train station connection reward."""

    def test_no_connection_no_reward(self, reward_calculator, game_state):
        """Test that no connection gives no reward."""
        reward = reward_calculator._check_station_connections(game_state, 0)
        assert reward == 0.0

    def test_first_connection_gives_reward(self, reward_calculator, board, game_state):
        """Test that first station connection gives reward."""
        # Manually add rail connecting to train station (node 8)
        # Find an edge connected to node 8
        for edge_id, edge in board.edges.items():
            if 8 in edge_id:
                edge.add_rail(0)  # Add rail for player 0
                break

        reward = reward_calculator._check_station_connections(game_state, 0)
        assert reward == 0.1  # Default station connection reward

    def test_second_connection_to_same_station_no_reward(self, reward_calculator, board, game_state):
        """Test that connecting to same station again gives no reward."""
        # Connect to station
        for edge_id, edge in board.edges.items():
            if 8 in edge_id:
                edge.add_rail(0)
                break

        # First call - gets reward
        reward1 = reward_calculator._check_station_connections(game_state, 0)
        assert reward1 == 0.1

        # Second call - no additional reward (already connected)
        reward2 = reward_calculator._check_station_connections(game_state, 0)
        assert reward2 == 0.0

    def test_connection_to_second_station_gives_reward(self, reward_calculator, board, game_state):
        """Test that connecting to second station gives additional reward."""
        # Connect to first station (node 8)
        for edge_id, edge in board.edges.items():
            if 8 in edge_id:
                edge.add_rail(0)
                break

        reward1 = reward_calculator._check_station_connections(game_state, 0)
        assert reward1 == 0.1

        # Connect to second station (node 27)
        for edge_id, edge in board.edges.items():
            if 27 in edge_id and not edge.has_player_rail(0):
                edge.add_rail(0)
                break

        reward2 = reward_calculator._check_station_connections(game_state, 0)
        assert reward2 == 0.1  # Another 0.1 for second station


class TestComputeReward:
    """Tests for the main compute_reward method."""

    def test_compute_reward_returns_float(self, reward_calculator, game_state):
        """Test that compute_reward returns a float."""
        prev_state = game_state.clone()
        reward = reward_calculator.compute_reward(
            game_state, prev_state, player_id=0, done=False
        )
        assert isinstance(reward, float)

    def test_compute_reward_detailed_returns_info(self, reward_calculator, game_state):
        """Test that compute_reward_detailed returns StepRewardInfo."""
        prev_state = game_state.clone()
        info = reward_calculator.compute_reward_detailed(
            game_state, prev_state, player_id=0, done=False
        )
        assert isinstance(info, StepRewardInfo)

    def test_compute_reward_matches_detailed_total(self, reward_calculator, game_state):
        """Test that compute_reward matches detailed total."""
        prev_state = game_state.clone()
        game_state.players[0].score = 5  # Add some score change

        reward = reward_calculator.compute_reward(
            game_state, prev_state, player_id=0, done=False
        )
        info = reward_calculator.compute_reward_detailed(
            game_state, prev_state, player_id=0, done=False
        )

        assert reward == info.total


class TestGetConnectedStations:
    """Tests for get_connected_stations utility method."""

    def test_returns_empty_for_new_player(self, reward_calculator):
        """Test that new players have no connected stations."""
        stations = reward_calculator.get_connected_stations(0)
        assert stations == set()

    def test_returns_copy(self, reward_calculator):
        """Test that get_connected_stations returns a copy."""
        reward_calculator._stations_connected[0] = {8, 27}
        stations = reward_calculator.get_connected_stations(0)

        # Modify the returned set
        stations.add(999)

        # Original should be unchanged
        assert 999 not in reward_calculator._stations_connected[0]
