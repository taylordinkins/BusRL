"""Action space mapping for the Bus RL environment.

Provides bidirectional mapping between flat action indices (for neural networks)
and structured Action objects (for the game engine).
"""

from __future__ import annotations

from typing import Optional

from core.game_state import GameState
from core.constants import ActionAreaType, BuildingType
from core.board import BoardGraph, EdgeId, make_edge_id
from engine.game_engine import Action, ActionType
from .config import ActionSpaceConfig, DEFAULT_ACTION_CONFIG


class ActionMapping:
    """Bidirectional mapping between flat action indices and Action objects.

    This class handles the conversion between:
    - Flat integer indices (0 to TOTAL_ACTIONS-1) used by neural networks
    - Structured Action objects used by the game engine

    The mapping is deterministic and consistent across all game states,
    allowing the RL agent to use a fixed action space.
    """

    def __init__(self, board: BoardGraph, config: ActionSpaceConfig = DEFAULT_ACTION_CONFIG):
        """Initialize action mapping with board topology.

        Args:
            board: The game board (needed for node/edge topology).
            config: Action space configuration.
        """
        self.config = config
        self.board = board

        # Build deterministic mappings
        self._node_ids = sorted(board.nodes.keys())
        self._edge_ids = sorted(board.edges.keys())

        # Pre-compute action mappings for efficiency
        self._building_actions = self._build_building_actions()
        self._edge_actions = self._build_edge_actions()
        self._vrroomm_actions = self._build_vrroomm_actions()

    def _build_building_actions(self) -> list[tuple[int, int, BuildingType]]:
        """Build list of (node_id, slot_idx, building_type) tuples."""
        actions = []
        for node_id in self._node_ids:
            for slot_idx in range(self.config.MAX_BUILDING_SLOTS_PER_NODE):
                for building_type in BuildingType:
                    actions.append((node_id, slot_idx, building_type))
        return actions

    def _build_edge_actions(self) -> list[EdgeId]:
        """Build list of edge IDs."""
        return self._edge_ids

    def _build_vrroomm_actions(self) -> list[tuple[int, int, int]]:
        """Build list of (passenger_id, node_id, slot_idx) tuples."""
        actions = []
        for passenger_id in range(self.config.MAX_PASSENGERS):
            for node_id in self._node_ids:
                for slot_idx in range(self.config.MAX_BUILDING_SLOTS_PER_NODE):
                    actions.append((passenger_id, node_id, slot_idx))
        return actions

    def index_to_action(self, action_idx: int, state: GameState) -> Action:
        """Convert flat action index to structured Action object.

        Args:
            action_idx: Flat action index (0 to TOTAL_ACTIONS-1).
            state: Current game state (needed for player_id).

        Returns:
            Structured Action object.

        Raises:
            ValueError: If action_idx is out of range.
        """
        if not 0 <= action_idx < self.config.total_actions:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.config.total_actions})")

        player_id = state.global_state.current_player_idx

        # PLACE_MARKER actions (0-6)
        if action_idx < self.config.place_marker_end:
            area_idx = action_idx - self.config.place_marker_start
            area_type = list(ActionAreaType)[area_idx]
            return Action(
                action_type=ActionType.PLACE_MARKER,
                player_id=player_id,
                params={"area_type": area_type.value}
            )

        # PASS action (7)
        elif action_idx == self.config.pass_idx:
            return Action(
                action_type=ActionType.PASS,
                player_id=player_id,
                params={}
            )

        # SETUP_BUILDING actions (8-223)
        elif self.config.setup_building_start <= action_idx < self.config.setup_building_end:
            idx = action_idx - self.config.setup_building_start
            node_id, slot_idx, building_type = self._building_actions[idx]
            return Action(
                action_type=ActionType.PLACE_BUILDING_SETUP,
                player_id=player_id,
                params={
                    "node_id": node_id,
                    "slot_index": slot_idx,
                    "building_type": building_type.value
                }
            )

        # SETUP_RAIL actions (224-293)
        elif self.config.setup_rail_start <= action_idx < self.config.setup_rail_end:
            idx = action_idx - self.config.setup_rail_start
            edge_id = self._edge_actions[idx]
            return Action(
                action_type=ActionType.PLACE_RAIL_SETUP,
                player_id=player_id,
                params={"edge_id": list(edge_id)}  # Convert tuple to list for JSON serialization
            )

        # LINE_EXPANSION actions (294-363)
        elif self.config.line_expansion_start <= action_idx < self.config.line_expansion_end:
            idx = action_idx - self.config.line_expansion_start
            edge_id = self._edge_actions[idx]
            return Action(
                action_type=ActionType.RESOLVE_LINE_EXPANSION,
                player_id=player_id,
                params={"edge_id": list(edge_id)}
            )

        # PASSENGERS actions (364-369)
        elif self.config.passengers_start <= action_idx < self.config.passengers_end:
            count_to_first_station = action_idx - self.config.passengers_start
            return Action(
                action_type=ActionType.RESOLVE_PASSENGERS,
                player_id=player_id,
                params={"count_to_first_station": count_to_first_station}
            )

        # BUILDINGS actions (370-585)
        elif self.config.buildings_start <= action_idx < self.config.buildings_end:
            idx = action_idx - self.config.buildings_start
            node_id, slot_idx, building_type = self._building_actions[idx]
            return Action(
                action_type=ActionType.RESOLVE_BUILDINGS,
                player_id=player_id,
                params={
                    "node_id": node_id,
                    "slot_index": slot_idx,
                    "building_type": building_type.value
                }
            )

        # TIME_CLOCK actions (586-587)
        elif self.config.time_clock_start <= action_idx < self.config.time_clock_end:
            is_advance = (action_idx - self.config.time_clock_start) == 0
            return Action(
                action_type=ActionType.RESOLVE_TIME_CLOCK,
                player_id=player_id,
                params={"advance": is_advance}
            )

        # VRROOMM actions (588-1667)
        elif self.config.vrroomm_start <= action_idx < self.config.vrroomm_end:
            idx = action_idx - self.config.vrroomm_start
            passenger_id, node_id, slot_idx = self._vrroomm_actions[idx]
            return Action(
                action_type=ActionType.RESOLVE_VRROOMM,
                player_id=player_id,
                params={
                    "passenger_id": passenger_id,
                    "to_node": node_id,
                    "building_slot_index": slot_idx
                }
            )

        # SKIP_DELIVERY action (1668)
        elif action_idx == self.config.skip_delivery_idx:
            return Action(
                action_type=ActionType.RESOLVE_VRROOMM,
                player_id=player_id,
                params={"skip": True}
            )

        # NOOP action (1669)
        elif action_idx == self.config.noop_idx:
            # NOOP is used for phases that auto-advance
            return Action(
                action_type=ActionType.PASS,  # Use PASS as placeholder
                player_id=player_id,
                params={"noop": True}
            )

        else:
            raise ValueError(f"Unhandled action index: {action_idx}")

    def action_to_index(self, action: Action) -> int:
        """Convert structured Action object to flat action index.

        Args:
            action: Structured Action object.

        Returns:
            Flat action index (0 to TOTAL_ACTIONS-1).

        Raises:
            ValueError: If action cannot be mapped to an index.
        """
        action_type = action.action_type
        params = action.params

        # PLACE_MARKER
        if action_type == ActionType.PLACE_MARKER:
            area_type_str = params["area_type"]
            area_type = ActionAreaType(area_type_str)
            area_idx = list(ActionAreaType).index(area_type)
            return self.config.place_marker_start + area_idx

        # PASS
        elif action_type == ActionType.PASS:
            if params.get("noop", False):
                return self.config.noop_idx
            return self.config.pass_idx

        # SETUP_BUILDING
        elif action_type == ActionType.PLACE_BUILDING_SETUP:
            node_id = params["node_id"]
            slot_idx = params["slot_index"]
            building_type = BuildingType(params["building_type"])
            target = (node_id, slot_idx, building_type)
            try:
                idx = self._building_actions.index(target)
                return self.config.setup_building_start + idx
            except ValueError:
                raise ValueError(f"Invalid building action: {target}")

        # SETUP_RAIL
        elif action_type == ActionType.PLACE_RAIL_SETUP:
            edge_id_list = params["edge_id"]
            edge_id = make_edge_id(edge_id_list[0], edge_id_list[1])
            try:
                idx = self._edge_actions.index(edge_id)
                return self.config.setup_rail_start + idx
            except ValueError:
                raise ValueError(f"Invalid rail edge: {edge_id}")

        # LINE_EXPANSION
        elif action_type == ActionType.RESOLVE_LINE_EXPANSION:
            edge_id_list = params["edge_id"]
            edge_id = make_edge_id(edge_id_list[0], edge_id_list[1])
            try:
                idx = self._edge_actions.index(edge_id)
                return self.config.line_expansion_start + idx
            except ValueError:
                raise ValueError(f"Invalid line expansion edge: {edge_id}")

        # PASSENGERS
        elif action_type == ActionType.RESOLVE_PASSENGERS:
            count = params["count_to_first_station"]
            if not 0 <= count < self.config.passengers_count:
                raise ValueError(f"Invalid passenger count: {count}")
            return self.config.passengers_start + count

        # BUILDINGS
        elif action_type == ActionType.RESOLVE_BUILDINGS:
            node_id = params["node_id"]
            slot_idx = params["slot_index"]
            building_type = BuildingType(params["building_type"])
            target = (node_id, slot_idx, building_type)
            try:
                idx = self._building_actions.index(target)
                return self.config.buildings_start + idx
            except ValueError:
                raise ValueError(f"Invalid building action: {target}")

        # TIME_CLOCK
        elif action_type == ActionType.RESOLVE_TIME_CLOCK:
            is_advance = params.get("advance", True)
            offset = 0 if is_advance else 1
            return self.config.time_clock_start + offset

        # VRROOMM
        elif action_type == ActionType.RESOLVE_VRROOMM:
            if params.get("skip", False):
                return self.config.skip_delivery_idx

            passenger_id = params["passenger_id"]
            node_id = params["to_node"]
            slot_idx = params["building_slot_index"]
            target = (passenger_id, node_id, slot_idx)
            try:
                idx = self._vrroomm_actions.index(target)
                return self.config.vrroomm_start + idx
            except ValueError:
                raise ValueError(f"Invalid vrroomm action: {target}")

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def get_action_range(self, action_type: ActionType) -> tuple[int, int]:
        """Get the index range for a specific action type.

        Args:
            action_type: The action type.

        Returns:
            Tuple of (start_idx, end_idx) for the action type.
        """
        if action_type == ActionType.PLACE_MARKER:
            return (self.config.place_marker_start, self.config.place_marker_end)
        elif action_type == ActionType.PASS:
            return (self.config.pass_idx, self.config.pass_idx + 1)
        elif action_type == ActionType.PLACE_BUILDING_SETUP:
            return (self.config.setup_building_start, self.config.setup_building_end)
        elif action_type == ActionType.PLACE_RAIL_SETUP:
            return (self.config.setup_rail_start, self.config.setup_rail_end)
        elif action_type == ActionType.RESOLVE_LINE_EXPANSION:
            return (self.config.line_expansion_start, self.config.line_expansion_end)
        elif action_type == ActionType.RESOLVE_PASSENGERS:
            return (self.config.passengers_start, self.config.passengers_end)
        elif action_type == ActionType.RESOLVE_BUILDINGS:
            return (self.config.buildings_start, self.config.buildings_end)
        elif action_type == ActionType.RESOLVE_TIME_CLOCK:
            return (self.config.time_clock_start, self.config.time_clock_end)
        elif action_type == ActionType.RESOLVE_VRROOMM:
            return (self.config.vrroomm_start, self.config.skip_delivery_idx + 1)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    @property
    def total_actions(self) -> int:
        """Total number of actions in the space."""
        return self.config.total_actions
