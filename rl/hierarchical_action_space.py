"""Hierarchical action mapping for phase-aware PPO.

Provides deterministic, seed-independent action catalogs per head and
helpers to convert engine/resolver actions into head-local indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from core.constants import Phase, ActionAreaType, BuildingType, TOTAL_PASSENGERS
from core.board import BoardGraph, EdgeId, make_edge_id
from engine.game_engine import Action, ActionType
from engine.resolvers.time_clock import TimeClockAction

from .config import BoardConfig


class HeadId(Enum):
    """Decision head identifiers (10 total)."""

    SETUP_BUILDINGS = 0
    SETUP_RAILS_FORWARD = 1
    SETUP_RAILS_REVERSE = 2
    CHOOSING_ACTIONS = 3
    RESOLVE_LINE_EXPANSION = 4
    RESOLVE_PASSENGERS = 5
    RESOLVE_BUILDINGS = 6
    RESOLVE_TIME_CLOCK = 7
    RESOLVE_VRROOMM_PASSENGER = 8
    RESOLVE_VRROOMM_DEST = 9


VRROOMM_SKIP = "SKIP"


def get_head_id(
    phase: Phase,
    resolution_area: Optional[ActionAreaType],
    vrroomm_stage: int,
) -> Optional[HeadId]:
    """Get head id for the current decision context.

    Returns None for automatic phases/areas (e.g., BUSES, STARTING_PLAYER).
    """
    if phase == Phase.SETUP_BUILDINGS:
        return HeadId.SETUP_BUILDINGS
    if phase == Phase.SETUP_RAILS_FORWARD:
        return HeadId.SETUP_RAILS_FORWARD
    if phase == Phase.SETUP_RAILS_REVERSE:
        return HeadId.SETUP_RAILS_REVERSE
    if phase == Phase.CHOOSING_ACTIONS:
        return HeadId.CHOOSING_ACTIONS

    if phase != Phase.RESOLVING_ACTIONS or resolution_area is None:
        return None

    if resolution_area == ActionAreaType.LINE_EXPANSION:
        return HeadId.RESOLVE_LINE_EXPANSION
    if resolution_area == ActionAreaType.PASSENGERS:
        return HeadId.RESOLVE_PASSENGERS
    if resolution_area == ActionAreaType.BUILDINGS:
        return HeadId.RESOLVE_BUILDINGS
    if resolution_area == ActionAreaType.TIME_CLOCK:
        return HeadId.RESOLVE_TIME_CLOCK
    if resolution_area == ActionAreaType.VRROOMM:
        if vrroomm_stage == 1:
            return HeadId.RESOLVE_VRROOMM_PASSENGER
        if vrroomm_stage == 2:
            return HeadId.RESOLVE_VRROOMM_DEST
        return None

    # BUSES / STARTING_PLAYER or any other auto areas
    return None


@dataclass(frozen=True)
class HeadCatalog:
    head_id: HeadId
    actions: list[Any]


class HierarchicalActionMapping:
    """Deterministic action catalogs and index mapping per head."""

    def __init__(
        self,
        board: BoardGraph,
        max_building_slots_per_node: int = BoardConfig.MAX_BUILDING_SLOTS_PER_NODE,
    ) -> None:
        self.board = board
        self.max_building_slots_per_node = max_building_slots_per_node

        self._node_ids = sorted(board.nodes.keys())
        self._edge_ids = sorted(board.edges.keys())

        self._building_actions = self._build_building_actions()
        self._setup_rail_actions = list(self._edge_ids)
        self._choosing_actions = self._build_choosing_actions()
        self._line_expansion_actions = self._build_line_expansion_actions()
        self._passenger_distributions = list(range(6))
        self._time_clock_actions = [True, False]  # True=advance, False=stop
        self._vrroomm_passenger_actions = list(range(TOTAL_PASSENGERS)) + [VRROOMM_SKIP]
        self._vrroomm_dest_actions = self._build_vrroomm_destinations()

        self._catalogs: dict[HeadId, list[Any]] = {
            HeadId.SETUP_BUILDINGS: self._building_actions,
            HeadId.SETUP_RAILS_FORWARD: self._setup_rail_actions,
            HeadId.SETUP_RAILS_REVERSE: self._setup_rail_actions,
            HeadId.CHOOSING_ACTIONS: self._choosing_actions,
            HeadId.RESOLVE_LINE_EXPANSION: self._line_expansion_actions,
            HeadId.RESOLVE_PASSENGERS: self._passenger_distributions,
            HeadId.RESOLVE_BUILDINGS: self._building_actions,
            HeadId.RESOLVE_TIME_CLOCK: self._time_clock_actions,
            HeadId.RESOLVE_VRROOMM_PASSENGER: self._vrroomm_passenger_actions,
            HeadId.RESOLVE_VRROOMM_DEST: self._vrroomm_dest_actions,
        }

        # Precompute index maps for fast action->index lookup
        self._index_maps: dict[HeadId, dict[Any, int]] = {
            head: {action: idx for idx, action in enumerate(actions)}
            for head, actions in self._catalogs.items()
        }

    def get_catalog(self, head_id: HeadId) -> list[Any]:
        return list(self._catalogs[head_id])

    def head_size(self, head_id: HeadId) -> int:
        return len(self._catalogs[head_id])

    def index_to_action(self, head_id: HeadId, index: int) -> Any:
        actions = self._catalogs[head_id]
        if not 0 <= index < len(actions):
            raise ValueError(f"Index {index} out of range for head {head_id}")
        return actions[index]

    def action_to_index(self, head_id: HeadId, action: Any) -> int:
        action_key = self.action_to_key(head_id, action)
        try:
            return self._index_maps[head_id][action_key]
        except KeyError as e:
            raise ValueError(f"Action {action_key} not in catalog for head {head_id}") from e

    def action_to_key(self, head_id: HeadId, action: Any) -> Any:
        """Normalize an engine/resolver action into a catalog key."""
        if isinstance(action, Action):
            return self._engine_action_to_key(head_id, action)
        if isinstance(action, dict):
            return self._resolver_action_to_key(head_id, action)
        return action

    def _engine_action_to_key(self, head_id: HeadId, action: Action) -> Any:
        if head_id == HeadId.CHOOSING_ACTIONS:
            if action.action_type == ActionType.PASS:
                return ("PASS", None)
            if action.action_type == ActionType.PLACE_MARKER:
                area = ActionAreaType(action.params["area_type"])
                return ("PLACE_MARKER", area)
            raise ValueError(f"Unexpected action for choosing head: {action.action_type}")

        if head_id in (HeadId.SETUP_BUILDINGS, HeadId.RESOLVE_BUILDINGS):
            if action.action_type not in (ActionType.PLACE_BUILDING_SETUP, ActionType.RESOLVE_BUILDINGS):
                raise ValueError(f"Unexpected building action type: {action.action_type}")
            node_id = int(action.params["node_id"])
            slot_idx = int(action.params["slot_index"])
            building_type = BuildingType(action.params["building_type"])
            return (node_id, slot_idx, building_type)

        if head_id in (HeadId.SETUP_RAILS_FORWARD, HeadId.SETUP_RAILS_REVERSE):
            if action.action_type != ActionType.PLACE_RAIL_SETUP:
                raise ValueError(f"Unexpected rail setup action type: {action.action_type}")
            edge_id_list = action.params["edge_id"]
            edge_id = make_edge_id(edge_id_list[0], edge_id_list[1])
            return edge_id

        raise ValueError(f"Unsupported engine action for head {head_id}")

    def _resolver_action_to_key(self, head_id: HeadId, action: dict) -> Any:
        if head_id == HeadId.RESOLVE_LINE_EXPANSION:
            edge = action.get("edge_id")
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                edge_id = make_edge_id(edge[0], edge[1])
            else:
                edge_id = edge
            return (edge_id, int(action["from_endpoint"]))

        if head_id == HeadId.RESOLVE_PASSENGERS:
            if "count_to_first_station" in action:
                return int(action["count_to_first_station"])
            if "distribution" in action:
                return self._distribution_to_count(action["distribution"])
            raise ValueError("Passengers action missing distribution/count")

        if head_id == HeadId.RESOLVE_BUILDINGS:
            node_id = int(action["node_id"])
            slot_idx = int(action["slot_index"])
            building_type = action["building_type"]
            if not isinstance(building_type, BuildingType):
                building_type = BuildingType(building_type)
            return (node_id, slot_idx, building_type)

        if head_id == HeadId.RESOLVE_TIME_CLOCK:
            action_val = action.get("action")
            if isinstance(action_val, TimeClockAction):
                return action_val == TimeClockAction.ADVANCE_CLOCK
            if isinstance(action_val, str):
                return action_val == TimeClockAction.ADVANCE_CLOCK.value
            # Default to advance=True if unknown
            return True

        if head_id == HeadId.RESOLVE_VRROOMM_PASSENGER:
            if action.get("skip", False):
                return VRROOMM_SKIP
            return int(action["passenger_id"])

        if head_id == HeadId.RESOLVE_VRROOMM_DEST:
            return (int(action["to_node"]), int(action["building_slot_index"]))

        raise ValueError(f"Unsupported resolver action for head {head_id}")

    def _distribution_to_count(self, distribution: dict[int, int]) -> int:
        stations = sorted(n.node_id for n in self.board.get_train_stations())
        if not stations:
            return 0
        return int(distribution.get(stations[0], 0))

    def _build_building_actions(self) -> list[tuple[int, int, BuildingType]]:
        actions: list[tuple[int, int, BuildingType]] = []
        for node_id in self._node_ids:
            for slot_idx in range(self.max_building_slots_per_node):
                for building_type in BuildingType:
                    actions.append((node_id, slot_idx, building_type))
        return actions

    def _build_line_expansion_actions(self) -> list[tuple[EdgeId, int]]:
        actions: list[tuple[EdgeId, int]] = []
        for edge_id in self._edge_ids:
            a, b = edge_id
            actions.append((edge_id, a))
            actions.append((edge_id, b))
        return actions

    def _build_choosing_actions(self) -> list[tuple[str, Optional[ActionAreaType]]]:
        actions: list[tuple[str, Optional[ActionAreaType]]] = []
        for area in list(ActionAreaType):
            actions.append(("PLACE_MARKER", area))
        actions.append(("PASS", None))
        return actions

    def _build_vrroomm_destinations(self) -> list[tuple[int, int]]:
        dests: list[tuple[int, int]] = []
        for node_id in self._node_ids:
            node = self.board.nodes[node_id]
            for slot_idx in range(len(node.building_slots)):
                dests.append((node_id, slot_idx))
        return dests
