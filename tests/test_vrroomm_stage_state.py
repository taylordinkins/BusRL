import os
import sys

import pytest

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_default_board
from core.game_state import GameState
from rl.vrroomm_stage import VrroommStageState
from rl.bus_env import BusEnv


def _make_state():
    board = load_default_board()
    state = GameState.create_initial_state(board, num_players=3)
    passenger = state.passenger_manager.create_passenger(8)
    state.board.get_node(8).add_passenger(passenger.passenger_id)
    return state, passenger.passenger_id


def test_vrroomm_stage_transitions_and_action_build():
    state, passenger_id = _make_state()
    stage = VrroommStageState()

    stage.enter()
    assert stage.stage == 1

    stage.select_passenger(passenger_id)
    assert stage.stage == 2

    action = stage.build_delivery_action(state, to_node=8, building_slot_index=0)
    assert action["passenger_id"] == passenger_id
    assert action["from_node"] == 8
    assert action["to_node"] == 8
    assert action["building_slot_index"] == 0
    assert stage.stage == 1  # ready for next delivery

    stage.complete()
    assert stage.stage == 0


def test_vrroomm_skip_resets():
    state, passenger_id = _make_state()
    stage = VrroommStageState()
    stage.enter()
    stage.select_passenger(passenger_id)
    stage.skip()
    assert stage.stage == 0


def test_bus_env_vrroomm_passenger_tiebreak():
    env = BusEnv(num_players=3)
    env.reset()
    state = env.get_state()

    # Create two passengers at the same origin node
    p1 = state.passenger_manager.create_passenger(8)
    state.board.get_node(8).add_passenger(p1.passenger_id)
    p2 = state.passenger_manager.create_passenger(8)
    state.board.get_node(8).add_passenger(p2.passenger_id)

    env._enter_vrroomm()
    valid_ids = {p1.passenger_id, p2.passenger_id}
    chosen = env._select_vrroomm_passenger_with_tiebreak(
        max(p1.passenger_id, p2.passenger_id),
        valid_ids,
    )
    assert chosen == min(p1.passenger_id, p2.passenger_id)

    # Ensure delivery action uses the chosen (lowest) passenger
    action = env._build_vrroomm_delivery_action(to_node=8, building_slot_index=0)
    assert action["passenger_id"] == chosen
