"""Tests for rl/observation.py - Observation encoding for RL."""

import pytest
import numpy as np

from rl.observation import ObservationEncoder
from rl.config import ObservationConfig, DEFAULT_OBS_CONFIG
from core.game_state import GameState
from core.constants import Phase, BuildingType, ActionAreaType
from data.loader import load_board


@pytest.fixture
def board():
    """Load the default board for testing."""
    from pathlib import Path
    board_path = Path(__file__).parent.parent / "data" / "default_board.json"
    return load_board(board_path)


@pytest.fixture
def initial_state(board):
    """Create an initial game state for testing."""
    return GameState.create_initial_state(board, num_players=4)


@pytest.fixture
def encoder():
    """Create an observation encoder with default config."""
    return ObservationEncoder()


class TestObservationEncoderInit:
    """Tests for ObservationEncoder initialization."""

    def test_default_config(self):
        """Test encoder uses default config if none provided."""
        encoder = ObservationEncoder()
        assert encoder.config == DEFAULT_OBS_CONFIG

    def test_custom_config(self):
        """Test encoder accepts custom config."""
        config = ObservationConfig()
        encoder = ObservationEncoder(config)
        assert encoder.config is config

    def test_observation_dim(self, encoder):
        """Test observation_dim property matches config."""
        assert encoder.observation_dim == encoder.config.total_observation_dim


class TestObservationEncoderEncode:
    """Tests for the encode method."""

    def test_encode_returns_numpy_array(self, encoder, initial_state):
        """Test that encode returns a numpy array."""
        obs = encoder.encode(initial_state)
        assert isinstance(obs, np.ndarray)

    def test_encode_correct_shape(self, encoder, initial_state):
        """Test that encoded observation has correct shape."""
        obs = encoder.encode(initial_state)
        expected_shape = (encoder.config.total_observation_dim,)
        assert obs.shape == expected_shape

    def test_encode_correct_dtype(self, encoder, initial_state):
        """Test that encoded observation has float32 dtype."""
        obs = encoder.encode(initial_state)
        assert obs.dtype == np.float32

    def test_encode_values_in_range(self, encoder, initial_state):
        """Test that all values are in [0, 1] range."""
        obs = encoder.encode(initial_state)
        assert np.all(obs >= 0.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 1.0), f"Max value: {obs.max()}"

    def test_encode_with_explicit_player_id(self, encoder, initial_state):
        """Test encoding from a specific player's perspective."""
        obs0 = encoder.encode(initial_state, current_player_id=0)
        obs1 = encoder.encode(initial_state, current_player_id=1)

        # Different perspectives should give different observations
        # (at least the player features should differ)
        assert not np.allclose(obs0, obs1)

    def test_encode_without_player_id_uses_current(self, encoder, initial_state):
        """Test that encode without player_id uses current player from state."""
        # Ensure current player is 0
        initial_state.global_state.current_player_idx = 0
        obs_implicit = encoder.encode(initial_state)
        obs_explicit = encoder.encode(initial_state, current_player_id=0)

        assert np.allclose(obs_implicit, obs_explicit)


class TestObservationEncoderNodeFeatures:
    """Tests for node feature encoding."""

    def test_node_features_size(self, encoder, initial_state):
        """Test that node features occupy the expected portion of observation."""
        # Encode twice to initialize mappings
        obs = encoder.encode(initial_state)

        # Node features should be at the start of the observation
        node_section = obs[:encoder.config.node_features_size]
        assert len(node_section) == encoder.config.MAX_NODES * encoder.config.NODE_FEATURE_DIM

    def test_train_station_encoding(self, encoder, initial_state):
        """Test that train stations are correctly encoded."""
        obs = encoder.encode(initial_state)

        # Train stations are nodes 8 and 27
        # is_train_station is feature index 2 (after position_x, position_y)
        feature_dim = encoder.config.NODE_FEATURE_DIM

        # Get node index for train station 8
        node_idx_8 = encoder._node_to_idx.get(8)
        if node_idx_8 is not None:
            base = node_idx_8 * feature_dim
            is_train_station_idx = 2
            assert obs[base + is_train_station_idx] == 1.0

    def test_central_park_encoding(self, encoder, initial_state):
        """Test that central parks are correctly encoded."""
        obs = encoder.encode(initial_state)

        # Central parks are nodes 11, 14, 15, 20
        # is_central_park is feature index 3
        feature_dim = encoder.config.NODE_FEATURE_DIM

        for park_id in [11, 14, 15, 20]:
            node_idx = encoder._node_to_idx.get(park_id)
            if node_idx is not None:
                base = node_idx * feature_dim
                is_central_park_idx = 3
                assert obs[base + is_central_park_idx] == 1.0, f"Central park {park_id} not encoded correctly"


class TestObservationEncoderEdgeFeatures:
    """Tests for edge feature encoding."""

    def test_edge_features_size(self, encoder, initial_state):
        """Test that edge features occupy the expected portion of observation."""
        obs = encoder.encode(initial_state)

        start = encoder.config.node_features_size
        end = start + encoder.config.edge_features_size
        edge_section = obs[start:end]
        assert len(edge_section) == encoder.config.MAX_EDGES * encoder.config.EDGE_FEATURE_DIM

    def test_empty_edges_at_start(self, encoder, initial_state):
        """Test that all edges are empty at game start."""
        obs = encoder.encode(initial_state)

        # During setup, all edges should be empty
        # is_empty is feature index 0 within edge features
        feature_dim = encoder.config.EDGE_FEATURE_DIM
        start = encoder.config.node_features_size

        # Check first few edges
        for edge_idx in range(min(5, encoder.config.MAX_EDGES)):
            base = start + edge_idx * feature_dim
            is_empty_idx = 0
            assert obs[base + is_empty_idx] == 1.0, f"Edge {edge_idx} should be empty at start"


class TestObservationEncoderPlayerFeatures:
    """Tests for player feature encoding."""

    def test_player_features_size(self, encoder, initial_state):
        """Test that player features occupy the expected portion of observation."""
        obs = encoder.encode(initial_state)

        start = encoder.config.node_features_size + encoder.config.edge_features_size
        end = start + encoder.config.player_features_size
        player_section = obs[start:end]
        assert len(player_section) == encoder.config.MAX_PLAYERS * encoder.config.PLAYER_FEATURE_DIM

    def test_current_player_is_first(self, encoder, initial_state):
        """Test that current player is always encoded first."""
        # Encode from player 2's perspective
        initial_state.global_state.current_player_idx = 2
        obs = encoder.encode(initial_state, current_player_id=2)

        # is_current_player is feature index 0 within player features
        feature_dim = encoder.config.PLAYER_FEATURE_DIM
        start = encoder.config.node_features_size + encoder.config.edge_features_size

        # First player slot should have is_current_player = 1
        assert obs[start + 0] == 1.0

        # Second player slot should have is_current_player = 0
        assert obs[start + feature_dim + 0] == 0.0

    def test_player_resources_normalized(self, encoder, initial_state):
        """Test that player resources are properly normalized."""
        obs = encoder.encode(initial_state)

        feature_dim = encoder.config.PLAYER_FEATURE_DIM
        start = encoder.config.node_features_size + encoder.config.edge_features_size

        # action_markers_remaining is feature index 2
        # At game start, players have 20 markers, normalized by 20 = 1.0
        markers_idx = 2
        assert obs[start + markers_idx] == 1.0

        # buses is feature index 4
        # At game start, players have 1 bus, normalized by 5 = 0.2
        buses_idx = 4
        assert obs[start + buses_idx] == pytest.approx(0.2, rel=0.01)


class TestObservationEncoderActionBoard:
    """Tests for action board encoding."""

    def test_action_board_size(self, encoder, initial_state):
        """Test that action board features occupy the expected portion."""
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
        )
        end = start + encoder.config.action_board_size
        ab_section = obs[start:end]
        expected_size = (
            encoder.config.ACTION_AREAS
            * encoder.config.MAX_SLOTS_PER_AREA
            * encoder.config.SLOT_FEATURE_DIM
        )
        assert len(ab_section) == expected_size

    def test_action_board_empty_at_start(self, encoder, initial_state):
        """Test that action board is empty at game start."""
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
        )

        # All slots should be unoccupied (is_occupied = 0)
        feature_dim = encoder.config.SLOT_FEATURE_DIM
        for area_idx in range(encoder.config.ACTION_AREAS):
            for slot_idx in range(encoder.config.MAX_SLOTS_PER_AREA):
                base = start + (area_idx * encoder.config.MAX_SLOTS_PER_AREA + slot_idx) * feature_dim
                is_occupied_idx = 0
                assert obs[base + is_occupied_idx] == 0.0


class TestObservationEncoderPassengers:
    """Tests for passenger feature encoding."""

    def test_passenger_features_size(self, encoder, initial_state):
        """Test that passenger features occupy the expected portion."""
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
            + encoder.config.action_board_size
        )
        end = start + encoder.config.passenger_features_size
        passenger_section = obs[start:end]
        expected_size = encoder.config.MAX_PASSENGERS * encoder.config.PASSENGER_FEATURE_DIM
        assert len(passenger_section) == expected_size


class TestObservationEncoderGlobal:
    """Tests for global state encoding."""

    def test_global_features_size(self, encoder, initial_state):
        """Test that global features occupy the expected portion."""
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
            + encoder.config.action_board_size
            + encoder.config.passenger_features_size
        )
        end = start + encoder.config.global_features_size
        global_section = obs[start:end]
        assert len(global_section) == encoder.config.GLOBAL_FEATURE_DIM

    def test_phase_encoding(self, encoder, initial_state):
        """Test that phase is correctly one-hot encoded."""
        # Initial phase is SETUP_BUILDINGS
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
            + encoder.config.action_board_size
            + encoder.config.passenger_features_size
        )

        # Phase is first 7 values (one-hot)
        phase_section = obs[start:start + encoder.config.PHASES]

        # Should have exactly one 1.0
        assert np.sum(phase_section) == pytest.approx(1.0)
        assert np.max(phase_section) == 1.0

    def test_time_clock_encoding(self, encoder, initial_state):
        """Test that time clock position is correctly encoded."""
        obs = encoder.encode(initial_state)

        start = (
            encoder.config.node_features_size
            + encoder.config.edge_features_size
            + encoder.config.player_features_size
            + encoder.config.action_board_size
            + encoder.config.passenger_features_size
        )

        # Time clock is after phase (7) and round_number (1) = offset 8
        clock_start = start + encoder.config.PHASES + 1
        clock_section = obs[clock_start:clock_start + encoder.config.TIME_CLOCK_POSITIONS]

        # Should have exactly one 1.0
        assert np.sum(clock_section) == pytest.approx(1.0)

        # Initial position is HOUSE (index 0)
        assert clock_section[0] == 1.0


class TestObservationEncoderConsistency:
    """Tests for observation encoding consistency."""

    def test_deterministic_encoding(self, encoder, initial_state):
        """Test that encoding the same state gives the same result."""
        obs1 = encoder.encode(initial_state, current_player_id=0)
        obs2 = encoder.encode(initial_state, current_player_id=0)
        assert np.allclose(obs1, obs2)

    def test_cloned_state_same_observation(self, encoder, initial_state):
        """Test that cloned states produce identical observations."""
        cloned = initial_state.clone()
        obs1 = encoder.encode(initial_state)
        obs2 = encoder.encode(cloned)
        assert np.allclose(obs1, obs2)

    def test_different_states_different_observations(self, encoder, board):
        """Test that different states produce different observations."""
        state1 = GameState.create_initial_state(board, num_players=3)
        state2 = GameState.create_initial_state(board, num_players=4)

        obs1 = encoder.encode(state1)
        obs2 = encoder.encode(state2)

        # Different player counts should give different observations
        assert not np.allclose(obs1, obs2)


class TestObservationEncoderHelpers:
    """Tests for helper methods."""

    def test_get_observation_space_shape(self, encoder):
        """Test get_observation_space_shape returns correct tuple."""
        shape = encoder.get_observation_space_shape()
        assert isinstance(shape, tuple)
        assert len(shape) == 1
        assert shape[0] == encoder.config.total_observation_dim

    def test_get_observation_bounds(self, encoder):
        """Test get_observation_bounds returns correct bounds."""
        low, high = encoder.get_observation_bounds()
        assert low == 0.0
        assert high == 1.0

    def test_get_player_order_current_first(self, encoder):
        """Test that player order puts current player first."""
        order = encoder._get_player_order(current_player_id=2, num_players=4)
        assert order == [2, 3, 0, 1]

    def test_get_player_order_wraps_correctly(self, encoder):
        """Test that player order wraps around correctly."""
        order = encoder._get_player_order(current_player_id=3, num_players=5)
        assert order == [3, 4, 0, 1, 2]
