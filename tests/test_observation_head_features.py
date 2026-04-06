import os
import sys

import numpy as np

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_default_board
from core.game_state import GameState
from rl.config import ObservationConfig
from rl.hierarchical_action_space import HeadId
from rl.observation import ObservationEncoder


def test_head_and_vrroomm_stage_bits_appended():
    board = load_default_board()
    state = GameState.create_initial_state(board, num_players=3)
    config = ObservationConfig()
    encoder = ObservationEncoder(config)

    obs = encoder.encode(
        state,
        current_player_id=0,
        head_id=HeadId.RESOLVE_VRROOMM_DEST,
        vrroomm_stage=2,
    )

    tail = obs[-(config.HEAD_ID_DIM + config.VRROOMM_STAGE_DIM):]
    head_bits = tail[: config.HEAD_ID_DIM]
    stage_bits = tail[config.HEAD_ID_DIM :]

    assert obs.shape[0] == config.total_observation_dim
    assert np.isclose(head_bits.sum(), 1.0)
    assert head_bits[HeadId.RESOLVE_VRROOMM_DEST.value] == 1.0
    assert np.allclose(stage_bits, np.array([0.0, 1.0], dtype=np.float32))

