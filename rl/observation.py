"""Observation encoding for the Bus RL environment.

Encodes the complete GameState into a flat numpy array suitable for
neural network input. Uses self-relative player encoding where the
current player is always index 0.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from core.game_state import GameState
from core.board import BoardGraph, NodeId, EdgeId, make_edge_id
from core.constants import (
    Phase,
    BuildingType,
    ActionAreaType,
    Zone,
    ACTION_RESOLUTION_ORDER,
    TIME_CLOCK_ORDER,
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
    MAX_BUSES,
)
from .config import ObservationConfig, DEFAULT_OBS_CONFIG


class ObservationEncoder:
    """Encodes GameState into flat observation tensor.

    The observation is structured as follows:
    1. Node features [MAX_NODES x NODE_FEATURE_DIM]
    2. Edge features [MAX_EDGES x EDGE_FEATURE_DIM]
    3. Player features [MAX_PLAYERS x PLAYER_FEATURE_DIM]
    4. Action board [ACTION_AREAS x MAX_SLOTS_PER_AREA x SLOT_FEATURE_DIM]
    5. Passenger features [MAX_PASSENGERS x PASSENGER_FEATURE_DIM]
    6. Global state [GLOBAL_FEATURE_DIM]

    All features are normalized to [-1, 1] or [0, 1] range.
    Players are reordered so current player is always index 0.
    """

    def __init__(self, config: ObservationConfig = DEFAULT_OBS_CONFIG):
        """Initialize the encoder with configuration.

        Args:
            config: Observation configuration defining tensor dimensions.
        """
        self.config = config

        # Pre-compute mappings for consistent ordering
        self._node_to_idx: dict[NodeId, int] = {}
        self._edge_to_idx: dict[EdgeId, int] = {}
        self._initialized = False

    def _initialize_mappings(self, board: BoardGraph) -> None:
        """Initialize node and edge index mappings from board topology.

        Called once when first encoding a state to establish consistent
        ordering across all observations.
        """
        # Map nodes to indices (sorted by node_id for consistency)
        sorted_nodes = sorted(board.nodes.keys())
        self._node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}

        # Map edges to indices (sorted by edge_id for consistency)
        sorted_edges = sorted(board.edges.keys())
        self._edge_to_idx = {edge_id: idx for idx, edge_id in enumerate(sorted_edges)}

        self._initialized = True

    @property
    def observation_dim(self) -> int:
        """Total dimension of the flat observation tensor."""
        return self.config.total_observation_dim

    def encode(self, state: GameState, current_player_id: Optional[int] = None) -> np.ndarray:
        """Encode complete game state into flat observation tensor.

        Args:
            state: The GameState to encode.
            current_player_id: The player whose perspective to use.
                If None, uses state.global_state.current_player_idx.

        Returns:
            Flat numpy array of shape (total_observation_dim,) with dtype float32.
        """
        if not self._initialized:
            self._initialize_mappings(state.board)

        if current_player_id is None:
            current_player_id = state.global_state.current_player_idx

        # Allocate observation tensor
        obs = np.zeros(self.config.total_observation_dim, dtype=np.float32)

        # Encode each component
        offset = 0
        offset = self._encode_nodes(state, obs, offset)
        offset = self._encode_edges(state, current_player_id, obs, offset)
        offset = self._encode_players(state, current_player_id, obs, offset)
        offset = self._encode_action_board(state, current_player_id, obs, offset)
        offset = self._encode_passengers(state, obs, offset)
        offset = self._encode_global(state, current_player_id, obs, offset)

        return obs

    def _encode_nodes(self, state: GameState, obs: np.ndarray, offset: int) -> int:
        """Encode node features into observation tensor.

        Node features (18 per node):
        - position_x, position_y (2): normalized [0, 1]
        - is_train_station (1): binary
        - is_central_park (1): binary
        - passenger_count (1): normalized [0, 1]
        - num_building_slots (1): integer [0, 2]
        - slot_0: zone (4 one-hot) + building (4 one-hot) + occupied (1) = 9
        - slot_1: zone (4 one-hot) + building (4 one-hot) + occupied (1) = 9
        Total: 2 + 1 + 1 + 1 + 1 + 9 + 9 = 24... let's use 18 by condensing

        Revised (18 per node):
        - position_x, position_y (2)
        - is_train_station, is_central_park (2)
        - passenger_count (1): normalized
        - num_slots (1)
        - slot_0: zone_idx (1) + building_idx (1) + occupied (1) = 3
        - slot_1: zone_idx (1) + building_idx (1) + occupied (1) = 3
        - padding (6) to reach 18
        """
        feature_dim = self.config.NODE_FEATURE_DIM

        # Get board dimensions for normalization
        max_x = max(n.position[0] for n in state.board.nodes.values()) or 1.0
        max_y = max(n.position[1] for n in state.board.nodes.values()) or 1.0

        for node_id, node in state.board.nodes.items():
            idx = self._node_to_idx.get(node_id)
            if idx is None or idx >= self.config.MAX_NODES:
                continue

            base = offset + idx * feature_dim
            i = 0

            # Position (normalized to [0, 1])
            obs[base + i] = node.position[0] / max_x if max_x > 0 else 0.0
            i += 1
            obs[base + i] = node.position[1] / max_y if max_y > 0 else 0.0
            i += 1

            # Flags
            obs[base + i] = 1.0 if node.is_train_station else 0.0
            i += 1
            obs[base + i] = 1.0 if node.is_central_park else 0.0
            i += 1

            # Passenger count (normalized by max passengers)
            obs[base + i] = len(node.passenger_ids) / self.config.MAX_PASSENGERS
            i += 1

            # Number of building slots
            obs[base + i] = len(node.building_slots) / self.config.MAX_BUILDING_SLOTS_PER_NODE
            i += 1

            # Encode up to 2 building slots
            for slot_idx in range(self.config.MAX_BUILDING_SLOTS_PER_NODE):
                if slot_idx < len(node.building_slots):
                    slot = node.building_slots[slot_idx]
                    # Zone index (0-3 for A-D, normalized)
                    zone_idx = list(Zone).index(slot.zone)
                    obs[base + i] = zone_idx / 3.0
                    i += 1
                    # Building type (0=none, 1=house, 2=office, 3=pub, normalized)
                    if slot.building is None:
                        obs[base + i] = 0.0
                    else:
                        building_idx = list(BuildingType).index(slot.building) + 1
                        obs[base + i] = building_idx / 3.0
                    i += 1
                    # Occupied by passenger
                    obs[base + i] = 1.0 if slot.occupied_by_passenger_id is not None else 0.0
                    i += 1
                else:
                    # Empty slot placeholder (3 values)
                    obs[base + i] = 0.0
                    i += 1
                    obs[base + i] = 0.0
                    i += 1
                    obs[base + i] = 0.0
                    i += 1

            # Remaining features are padding (already 0)

        return offset + self.config.node_features_size

    def _encode_edges(
        self, state: GameState, current_player_id: int, obs: np.ndarray, offset: int
    ) -> int:
        """Encode edge features into observation tensor.

        Edge features (7 per edge):
        - is_empty (1): binary
        - has_current_player_rail (1): binary
        - player_rail_mask (5): binary for each player (self-relative order)
        """
        feature_dim = self.config.EDGE_FEATURE_DIM
        num_players = state.num_players()

        # Compute self-relative player order
        player_order = self._get_player_order(current_player_id, num_players)

        for edge_id, edge in state.board.edges.items():
            idx = self._edge_to_idx.get(edge_id)
            if idx is None or idx >= self.config.MAX_EDGES:
                continue

            base = offset + idx * feature_dim
            i = 0

            # Is empty
            obs[base + i] = 1.0 if edge.is_empty() else 0.0
            i += 1

            # Has current player rail
            obs[base + i] = 1.0 if edge.has_player_rail(current_player_id) else 0.0
            i += 1

            # Player rail mask (self-relative order)
            player_ids_with_rail = edge.get_player_ids()
            for rel_idx, actual_player_id in enumerate(player_order):
                if rel_idx >= self.config.MAX_PLAYERS:
                    break
                obs[base + i] = 1.0 if actual_player_id in player_ids_with_rail else 0.0
                i += 1

        return offset + self.config.edge_features_size

    def _encode_players(
        self, state: GameState, current_player_id: int, obs: np.ndarray, offset: int
    ) -> int:
        """Encode player features into observation tensor.

        Players are reordered so current player is always index 0.

        Player features (10 per player):
        - is_current_player (1): binary
        - is_starting_player (1): binary
        - action_markers_remaining (1): normalized [0, 1]
        - rail_segments_remaining (1): normalized [0, 1]
        - buses (1): normalized [0, 1]
        - score (1): normalized (divide by 20 as reasonable max)
        - time_stones (1): integer [0, 5] normalized
        - has_passed (1): binary
        - markers_placed_this_round (1): normalized
        - relative_position (1): turn order position from current player
        """
        feature_dim = self.config.PLAYER_FEATURE_DIM
        num_players = state.num_players()
        player_order = self._get_player_order(current_player_id, num_players)

        for rel_idx, actual_player_id in enumerate(player_order):
            if rel_idx >= self.config.MAX_PLAYERS:
                break

            player = state.players[actual_player_id]
            base = offset + rel_idx * feature_dim
            i = 0

            # Is current player (always 1 for index 0)
            obs[base + i] = 1.0 if actual_player_id == current_player_id else 0.0
            i += 1

            # Is starting player
            obs[base + i] = 1.0 if actual_player_id == state.global_state.starting_player_idx else 0.0
            i += 1

            # Action markers remaining (normalized)
            obs[base + i] = player.action_markers_remaining / TOTAL_ACTION_MARKERS
            i += 1

            # Rail segments remaining (normalized)
            obs[base + i] = player.rail_segments_remaining / TOTAL_RAIL_SEGMENTS
            i += 1

            # Buses (normalized)
            obs[base + i] = player.buses / MAX_BUSES
            i += 1

            # Score (normalized, assume max ~20 points)
            obs[base + i] = min(player.score / 20.0, 1.0)
            i += 1

            # Time stones (normalized)
            obs[base + i] = player.time_stones / 5.0
            i += 1

            # Has passed
            obs[base + i] = 1.0 if player.has_passed else 0.0
            i += 1

            # Markers placed this round (normalized)
            obs[base + i] = min(player.markers_placed_this_round / 6.0, 1.0)
            i += 1

            # Relative position (0 for current player, then clockwise)
            obs[base + i] = rel_idx / max(num_players - 1, 1)
            i += 1

        return offset + self.config.player_features_size

    def _encode_action_board(
        self, state: GameState, current_player_id: int, obs: np.ndarray, offset: int
    ) -> int:
        """Encode action board features into observation tensor.

        Action board features (4 per slot, 6 slots per area, 7 areas):
        - is_occupied (1): binary
        - occupying_player_relative_idx (1): normalized [-1=empty, 0-4=player]
        - is_current_player (1): binary
        - placement_order (1): normalized
        """
        feature_dim = self.config.SLOT_FEATURE_DIM
        num_players = state.num_players()
        player_order = self._get_player_order(current_player_id, num_players)

        # Create reverse mapping: actual_player_id -> relative_idx
        player_to_relative = {pid: idx for idx, pid in enumerate(player_order)}

        # Order areas by resolution order
        area_order = list(ActionAreaType)

        for area_idx, area_type in enumerate(area_order):
            if area_idx >= self.config.ACTION_AREAS:
                break

            area = state.action_board.areas.get(area_type)
            if area is None:
                continue

            # Get slots in label order (A, B, C, D, E, F)
            slot_labels = sorted(area.slots.keys())

            for slot_idx, label in enumerate(slot_labels):
                if slot_idx >= self.config.MAX_SLOTS_PER_AREA:
                    break

                slot = area.slots[label]
                base = offset + (area_idx * self.config.MAX_SLOTS_PER_AREA + slot_idx) * feature_dim
                i = 0

                # Is occupied
                is_occupied = slot.player_id is not None
                obs[base + i] = 1.0 if is_occupied else 0.0
                i += 1

                # Occupying player relative index
                if is_occupied:
                    rel_idx = player_to_relative.get(slot.player_id, -1)
                    obs[base + i] = (rel_idx + 1) / self.config.MAX_PLAYERS  # 0=empty, 0.2-1.0=players
                else:
                    obs[base + i] = 0.0
                i += 1

                # Is current player
                obs[base + i] = 1.0 if slot.player_id == current_player_id else 0.0
                i += 1

                # Placement order (normalized)
                if slot.placement_order is not None:
                    # Normalize by reasonable max (20 markers per player * 5 players)
                    obs[base + i] = min(slot.placement_order / 100.0, 1.0)
                else:
                    obs[base + i] = 0.0
                i += 1

        return offset + self.config.action_board_size

    def _encode_passengers(self, state: GameState, obs: np.ndarray, offset: int) -> int:
        """Encode passenger features into observation tensor.

        Passenger features (5 per passenger):
        - exists (1): binary
        - location_node_idx (1): normalized node index
        - is_at_train_station (1): binary
        - is_at_central_park (1): binary
        - is_at_matching_building (1): binary (matches time clock)
        """
        feature_dim = self.config.PASSENGER_FEATURE_DIM
        current_building = state.global_state.time_clock_position

        # Get passengers sorted by ID for consistent ordering
        sorted_passengers = sorted(
            state.passenger_manager.passengers.values(),
            key=lambda p: p.passenger_id
        )

        for idx, passenger in enumerate(sorted_passengers):
            if idx >= self.config.MAX_PASSENGERS:
                break

            base = offset + idx * feature_dim
            i = 0

            # Exists
            obs[base + i] = 1.0
            i += 1

            # Location node index (normalized)
            node_idx = self._node_to_idx.get(passenger.location, 0)
            obs[base + i] = node_idx / max(self.config.MAX_NODES - 1, 1)
            i += 1

            # Is at train station
            node = state.board.nodes.get(passenger.location)
            if node:
                obs[base + i] = 1.0 if node.is_train_station else 0.0
                i += 1
                obs[base + i] = 1.0 if node.is_central_park else 0.0
                i += 1
                # Is at matching building
                obs[base + i] = 1.0 if node.has_building_type(current_building) else 0.0
                i += 1
            else:
                i += 3  # Skip remaining features if node not found

        return offset + self.config.passenger_features_size

    def _encode_global(
        self, state: GameState, current_player_id: int, obs: np.ndarray, offset: int
    ) -> int:
        """Encode global state features into observation tensor.

        Global features (27 total):
        - phase (7): one-hot
        - round_number (1): normalized
        - time_clock_position (3): one-hot
        - time_stones_remaining (1): normalized
        - current_resolution_area (7): one-hot (during resolution)
        - current_resolution_slot (6): one-hot (during resolution)
        - max_buses (1): M#oB value normalized
        - all_players_passed (1): binary
        """
        base = offset
        i = 0

        # Phase (one-hot, 7 values)
        phase_idx = list(Phase).index(state.phase)
        for p_idx in range(self.config.PHASES):
            obs[base + i] = 1.0 if p_idx == phase_idx else 0.0
            i += 1

        # Round number (normalized, assume max ~15 rounds)
        obs[base + i] = min(state.global_state.round_number / 15.0, 1.0)
        i += 1

        # Time clock position (one-hot, 3 values)
        clock_idx = TIME_CLOCK_ORDER.index(state.global_state.time_clock_position)
        for c_idx in range(self.config.TIME_CLOCK_POSITIONS):
            obs[base + i] = 1.0 if c_idx == clock_idx else 0.0
            i += 1

        # Time stones remaining (normalized)
        obs[base + i] = state.global_state.time_stones_remaining / 5.0
        i += 1

        # Current resolution area (one-hot, 7 values)
        res_area = state.global_state.get_current_resolution_area()
        if res_area is not None:
            res_area_idx = ACTION_RESOLUTION_ORDER.index(res_area)
        else:
            res_area_idx = -1
        for r_idx in range(self.config.ACTION_AREAS):
            obs[base + i] = 1.0 if r_idx == res_area_idx else 0.0
            i += 1

        # Current resolution slot (one-hot, 6 values)
        res_slot_idx = state.global_state.current_resolution_slot_idx
        for s_idx in range(self.config.MAX_SLOTS_PER_AREA):
            obs[base + i] = 1.0 if s_idx == res_slot_idx else 0.0
            i += 1

        # Max buses (M#oB) normalized
        max_buses = max(p.buses for p in state.players)
        obs[base + i] = max_buses / MAX_BUSES
        i += 1

        # All players passed
        obs[base + i] = 1.0 if state.all_players_passed() else 0.0
        i += 1

        return offset + self.config.global_features_size

    def _get_player_order(self, current_player_id: int, num_players: int) -> list[int]:
        """Get player IDs in self-relative order (current player first, then clockwise)."""
        return [
            (current_player_id + i) % num_players
            for i in range(num_players)
        ]

    def get_observation_space_shape(self) -> tuple[int, ...]:
        """Return the shape of the observation space."""
        return (self.config.total_observation_dim,)

    def get_observation_bounds(self) -> tuple[float, float]:
        """Return the min and max bounds for observation values."""
        return (0.0, 1.0)
