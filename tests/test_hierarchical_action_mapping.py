import os
import sys

import pytest

# Ensure project root is on sys.path for imports like data.loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_default_board
from core.constants import Phase, ActionAreaType, BuildingType, TOTAL_PASSENGERS
from engine.game_engine import Action, ActionType
from core.board import make_edge_id

from rl.hierarchical_action_space import (
    HierarchicalActionMapping,
    HeadId,
    VRROOMM_SKIP,
    get_head_id,
)


def _mapping():
    board = load_default_board()
    return HierarchicalActionMapping(board)


def test_head_sizes_default_board():
    mapping = _mapping()
    assert mapping.head_size(HeadId.SETUP_BUILDINGS) == 216
    assert mapping.head_size(HeadId.SETUP_RAILS_FORWARD) == 70
    assert mapping.head_size(HeadId.SETUP_RAILS_REVERSE) == 70
    assert mapping.head_size(HeadId.CHOOSING_ACTIONS) == 8
    assert mapping.head_size(HeadId.RESOLVE_LINE_EXPANSION) == 140
    assert mapping.head_size(HeadId.RESOLVE_PASSENGERS) == 6
    assert mapping.head_size(HeadId.RESOLVE_BUILDINGS) == 216
    assert mapping.head_size(HeadId.RESOLVE_TIME_CLOCK) == 2
    assert mapping.head_size(HeadId.RESOLVE_VRROOMM_PASSENGER) == TOTAL_PASSENGERS + 1
    assert mapping.head_size(HeadId.RESOLVE_VRROOMM_DEST) == 47


def test_deterministic_catalogs():
    m1 = _mapping()
    m2 = _mapping()
    for head in HeadId:
        assert m1.get_catalog(head) == m2.get_catalog(head)


def test_line_expansion_actions_have_two_per_edge():
    mapping = _mapping()
    actions = mapping.get_catalog(HeadId.RESOLVE_LINE_EXPANSION)
    edges = sorted(load_default_board().edges.keys())
    assert len(actions) == len(edges) * 2
    edge = edges[0]
    assert (edge, edge[0]) in actions
    assert (edge, edge[1]) in actions


def test_vrroomm_destinations_sorted_by_node_and_slot():
    mapping = _mapping()
    dests = mapping.get_catalog(HeadId.RESOLVE_VRROOMM_DEST)
    assert dests == sorted(dests, key=lambda x: (x[0], x[1]))


def test_choosing_actions_map_to_areas_not_slots():
    mapping = _mapping()
    actions = mapping.get_catalog(HeadId.CHOOSING_ACTIONS)
    # Expect 7 area actions + PASS
    assert len(actions) == 8
    area_actions = [a for a in actions if a[0] == "PLACE_MARKER"]
    assert len(area_actions) == 7
    assert all(isinstance(a[1], ActionAreaType) for a in area_actions)


def test_action_to_index_roundtrip_setup_building():
    mapping = _mapping()
    action = Action(
        action_type=ActionType.PLACE_BUILDING_SETUP,
        player_id=0,
        params={"node_id": 0, "slot_index": 1, "building_type": BuildingType.HOUSE.value},
    )
    idx = mapping.action_to_index(HeadId.SETUP_BUILDINGS, action)
    key = mapping.index_to_action(HeadId.SETUP_BUILDINGS, idx)
    assert key == (0, 1, BuildingType.HOUSE)


def test_passengers_distribution_to_count():
    mapping = _mapping()
    # With default board, train stations are 8 and 27 (sorted)
    action = {"distribution": {8: 2, 27: 1}, "player_id": 0}
    key = mapping.action_to_key(HeadId.RESOLVE_PASSENGERS, action)
    assert key == 2


def test_vrroomm_passenger_skip():
    mapping = _mapping()
    idx = mapping.action_to_index(HeadId.RESOLVE_VRROOMM_PASSENGER, {"skip": True})
    assert mapping.index_to_action(HeadId.RESOLVE_VRROOMM_PASSENGER, idx) == VRROOMM_SKIP


def test_get_head_id_basic():
    assert get_head_id(Phase.CHOOSING_ACTIONS, None, 0) == HeadId.CHOOSING_ACTIONS
    assert get_head_id(Phase.SETUP_BUILDINGS, None, 0) == HeadId.SETUP_BUILDINGS
    assert get_head_id(Phase.RESOLVING_ACTIONS, ActionAreaType.LINE_EXPANSION, 0) == HeadId.RESOLVE_LINE_EXPANSION
    assert get_head_id(Phase.RESOLVING_ACTIONS, ActionAreaType.VRROOMM, 1) == HeadId.RESOLVE_VRROOMM_PASSENGER
    assert get_head_id(Phase.RESOLVING_ACTIONS, ActionAreaType.VRROOMM, 2) == HeadId.RESOLVE_VRROOMM_DEST
