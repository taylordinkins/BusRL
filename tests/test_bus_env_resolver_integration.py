import os
import sys

import numpy as np

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import Phase, ActionAreaType
from engine.resolvers.time_clock import TimeClockAction
from rl.bus_env import BusEnv
from rl.hierarchical_action_space import HeadId


def test_bus_env_time_clock_resolver_decision_flow():
    env = BusEnv(num_players=3)
    env.reset()
    state = env.get_state()

    # Force resolving phase with a Time Clock marker
    state.set_phase(Phase.RESOLVING_ACTIONS)
    state.global_state.current_resolution_area_idx = 0
    state.global_state.current_resolution_slot_idx = 0
    state.action_board.place_marker(ActionAreaType.TIME_CLOCK, 1)
    env._decision_cache = None

    mask = env.action_masks()
    decision = env._get_decision_context()
    assert decision["head_id"] == HeadId.RESOLVE_TIME_CLOCK
    assert state.global_state.current_player_idx == 1

    idx = env._hier_action_mapping.action_to_index(
        HeadId.RESOLVE_TIME_CLOCK,
        {"action": TimeClockAction.ADVANCE_CLOCK, "player_id": 1},
    )
    assert mask[idx]

    obs, reward, terminated, truncated, info = env.step(idx)
    assert not info.get("invalid_action", False)
    assert state.phase in (Phase.CLEANUP, Phase.CHOOSING_ACTIONS, Phase.RESOLVING_ACTIONS)

    if state.phase == Phase.RESOLVING_ACTIONS:
        next_mask = env.action_masks()
        assert np.any(next_mask)
