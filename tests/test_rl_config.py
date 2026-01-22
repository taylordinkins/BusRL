"""Tests for rl/config.py - RL configuration constants."""

import pytest
from rl.config import (
    BoardConfig,
    ObservationConfig,
    ActionSpaceConfig,
    RewardConfig,
    DEFAULT_OBS_CONFIG,
    DEFAULT_ACTION_CONFIG,
)


class TestBoardConfig:
    """Tests for BoardConfig."""

    def test_default_values(self):
        """Test that default board config has expected values."""
        config = BoardConfig()
        assert config.MAX_NODES == 36
        assert config.MAX_EDGES == 70
        assert config.MAX_BUILDING_SLOTS_PER_NODE == 2
        assert config.TRAIN_STATION_NODE_IDS == (8, 27)
        assert config.CENTRAL_PARK_NODE_IDS == (11, 14, 15, 20)

    def test_immutable(self):
        """Test that BoardConfig is immutable (frozen dataclass)."""
        config = BoardConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.MAX_NODES = 100


class TestObservationConfig:
    """Tests for ObservationConfig."""

    def test_default_values(self):
        """Test that default observation config has expected values."""
        config = ObservationConfig()
        assert config.MAX_NODES == 36
        assert config.MAX_EDGES == 70
        assert config.MAX_PLAYERS == 5
        assert config.MAX_PASSENGERS == 15
        assert config.ZONES == 4
        assert config.BUILDING_TYPES == 4
        assert config.PHASES == 7
        assert config.ACTION_AREAS == 7
        assert config.MAX_SLOTS_PER_AREA == 6

    def test_node_features_size(self):
        """Test node features size calculation."""
        config = ObservationConfig()
        expected = config.MAX_NODES * config.NODE_FEATURE_DIM
        assert config.node_features_size == expected
        assert config.node_features_size == 36 * 18

    def test_edge_features_size(self):
        """Test edge features size calculation."""
        config = ObservationConfig()
        expected = config.MAX_EDGES * config.EDGE_FEATURE_DIM
        assert config.edge_features_size == expected
        assert config.edge_features_size == 70 * 7

    def test_player_features_size(self):
        """Test player features size calculation."""
        config = ObservationConfig()
        expected = config.MAX_PLAYERS * config.PLAYER_FEATURE_DIM
        assert config.player_features_size == expected
        assert config.player_features_size == 5 * 10

    def test_action_board_size(self):
        """Test action board size calculation."""
        config = ObservationConfig()
        expected = config.ACTION_AREAS * config.MAX_SLOTS_PER_AREA * config.SLOT_FEATURE_DIM
        assert config.action_board_size == expected
        assert config.action_board_size == 7 * 6 * 4

    def test_passenger_features_size(self):
        """Test passenger features size calculation."""
        config = ObservationConfig()
        expected = config.MAX_PASSENGERS * config.PASSENGER_FEATURE_DIM
        assert config.passenger_features_size == expected
        assert config.passenger_features_size == 15 * 5

    def test_global_features_size(self):
        """Test global features size."""
        config = ObservationConfig()
        assert config.global_features_size == 27

    def test_total_observation_dim(self):
        """Test total observation dimension is sum of all components."""
        config = ObservationConfig()
        expected = (
            config.node_features_size
            + config.edge_features_size
            + config.player_features_size
            + config.action_board_size
            + config.passenger_features_size
            + config.global_features_size
        )
        assert config.total_observation_dim == expected

    def test_total_observation_dim_reasonable(self):
        """Test that total observation dim is within expected range."""
        config = ObservationConfig()
        # Should be approximately 1458 based on the plan
        assert 1400 <= config.total_observation_dim <= 1600


class TestActionSpaceConfig:
    """Tests for ActionSpaceConfig."""

    def test_default_values(self):
        """Test that default action space config has expected values."""
        config = ActionSpaceConfig()
        assert config.MAX_NODES == 36
        assert config.MAX_EDGES == 70
        assert config.MAX_PASSENGERS == 15
        assert config.NUM_ACTION_AREAS == 7
        assert config.NUM_BUILDING_TYPES == 3

    def test_place_marker_count(self):
        """Test PLACE_MARKER action count."""
        config = ActionSpaceConfig()
        assert config.place_marker_count == 7

    def test_setup_building_count(self):
        """Test SETUP_BUILDING action count: node x slot x type."""
        config = ActionSpaceConfig()
        expected = 36 * 2 * 3  # 216
        assert config.setup_building_count == expected
        assert config.setup_building_count == 216

    def test_setup_rail_count(self):
        """Test SETUP_RAIL action count."""
        config = ActionSpaceConfig()
        assert config.setup_rail_count == 70

    def test_vrroomm_count(self):
        """Test VRROOMM action count: passenger x node x slot."""
        config = ActionSpaceConfig()
        expected = 15 * 36 * 2
        assert config.vrroomm_count == expected

    def test_action_index_ranges_non_overlapping(self):
        """Test that action index ranges don't overlap."""
        config = ActionSpaceConfig()

        ranges = [
            (config.place_marker_start, config.place_marker_end),
            (config.pass_idx, config.pass_idx + 1),
            (config.setup_building_start, config.setup_building_end),
            (config.setup_rail_start, config.setup_rail_end),
            (config.line_expansion_start, config.line_expansion_end),
            (config.passengers_start, config.passengers_end),
            (config.buildings_start, config.buildings_end),
            (config.time_clock_start, config.time_clock_end),
            (config.vrroomm_start, config.vrroomm_end),
            (config.skip_delivery_idx, config.skip_delivery_idx + 1),
            (config.noop_idx, config.noop_idx + 1),
        ]

        # Check ranges are contiguous and non-overlapping
        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges):
                if i != j:
                    # Ranges should not overlap
                    assert end1 <= start2 or end2 <= start1, (
                        f"Ranges {i} and {j} overlap: [{start1}, {end1}) and [{start2}, {end2})"
                    )

    def test_action_index_ranges_contiguous(self):
        """Test that action index ranges are contiguous (no gaps)."""
        config = ActionSpaceConfig()

        # All ranges in order
        ranges = [
            config.place_marker_end,
            config.pass_idx + 1,
            config.setup_building_end,
            config.setup_rail_end,
            config.line_expansion_end,
            config.passengers_end,
            config.buildings_end,
            config.time_clock_end,
            config.vrroomm_end,
            config.skip_delivery_idx + 1,
            config.noop_idx + 1,
        ]

        starts = [
            config.place_marker_start,
            config.pass_idx,
            config.setup_building_start,
            config.setup_rail_start,
            config.line_expansion_start,
            config.passengers_start,
            config.buildings_start,
            config.time_clock_start,
            config.vrroomm_start,
            config.skip_delivery_idx,
            config.noop_idx,
        ]

        # Each end should equal the next start (contiguous)
        for i in range(len(ranges) - 1):
            assert ranges[i] == starts[i + 1], (
                f"Gap between range {i} (end={ranges[i]}) and range {i+1} (start={starts[i+1]})"
            )

    def test_total_actions(self):
        """Test total action count calculation."""
        config = ActionSpaceConfig()
        assert config.total_actions == config.noop_idx + 1

    def test_total_actions_approximately_expected(self):
        """Test that total actions is approximately 1670 as calculated."""
        config = ActionSpaceConfig()
        # Actual calculation: 7 + 1 + 216 + 70 + 70 + 6 + 216 + 2 + 1080 + 1 + 1 = 1670
        assert 1650 <= config.total_actions <= 1700
        assert config.total_actions == 1670


class TestRewardConfig:
    """Tests for RewardConfig."""

    def test_default_values(self):
        """Test that default reward config has expected values."""
        config = RewardConfig()
        assert config.delivery_reward == 1.0
        assert config.stolen_passenger_bonus == 0.1
        assert config.exclusive_delivery_bonus == 0.01
        assert config.station_connection_reward == 0.1
        assert config.time_stone_penalty == -0.01

    def test_mutable(self):
        """Test that RewardConfig is mutable (can be customized)."""
        config = RewardConfig()
        config.delivery_reward = 2.0
        assert config.delivery_reward == 2.0


class TestDefaultInstances:
    """Tests for default configuration instances."""

    def test_default_obs_config_exists(self):
        """Test that DEFAULT_OBS_CONFIG is available."""
        assert DEFAULT_OBS_CONFIG is not None
        assert isinstance(DEFAULT_OBS_CONFIG, ObservationConfig)

    def test_default_action_config_exists(self):
        """Test that DEFAULT_ACTION_CONFIG is available."""
        assert DEFAULT_ACTION_CONFIG is not None
        assert isinstance(DEFAULT_ACTION_CONFIG, ActionSpaceConfig)
