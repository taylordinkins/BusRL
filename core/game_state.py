"""Game state for the Bus game engine.

GameState is the single source of truth for the entire game.
It combines all components and provides methods for cloning,
serialization, and state hashing for RL rollouts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Any

from .constants import (
    Phase,
    BuildingType,
    ActionAreaType,
    MIN_PLAYERS,
    MAX_PLAYERS,
    TIME_CLOCK_ORDER,
    ACTION_RESOLUTION_ORDER,
)
from .board import BoardGraph, NodeId
from .player import Player
from .action_board import ActionBoard
from .components import PassengerManager


@dataclass
class GlobalState:
    """Global game state not tied to specific players or board positions.

    Attributes:
        round_number: Current round (1-indexed).
        current_player_idx: Index of the current player in the players list.
        starting_player_idx: Index of the starting player for this round.
        time_clock_position: Current position on the time clock (BuildingType).
        time_stones_remaining: Number of time stones not yet taken.
        current_resolution_area_idx: Index into ACTION_RESOLUTION_ORDER during resolution.
        current_resolution_slot_idx: Index of current slot being resolved within an area.
        game_ended: Whether the game has ended.
    """

    round_number: int = 1
    current_player_idx: int = 0
    starting_player_idx: int = 0
    time_clock_position: BuildingType = field(default_factory=lambda: TIME_CLOCK_ORDER[0])
    time_stones_remaining: int = 5  # Default for 4-5 players, 4 for 3 players
    current_resolution_area_idx: int = 0
    current_resolution_slot_idx: int = 0
    game_ended: bool = False

    def advance_time_clock(self) -> None:
        """Advance the time clock to the next building type (clockwise)."""
        current_idx = TIME_CLOCK_ORDER.index(self.time_clock_position)
        next_idx = (current_idx + 1) % len(TIME_CLOCK_ORDER)
        self.time_clock_position = TIME_CLOCK_ORDER[next_idx]

    def take_time_stone(self) -> bool:
        """Take a time stone if available.

        Returns:
            True if a time stone was taken, False if none remaining.
        """
        if self.time_stones_remaining > 0:
            self.time_stones_remaining -= 1
            return True
        return False

    def get_current_resolution_area(self) -> Optional[ActionAreaType]:
        """Get the action area currently being resolved.

        Returns:
            The ActionAreaType being resolved, or None if resolution is complete.
        """
        if self.current_resolution_area_idx >= len(ACTION_RESOLUTION_ORDER):
            return None
        return ACTION_RESOLUTION_ORDER[self.current_resolution_area_idx]

    def advance_resolution_slot(self) -> None:
        """Move to the next slot in the current resolution area."""
        self.current_resolution_slot_idx += 1

    def advance_resolution_area(self) -> None:
        """Move to the next action area in the resolution order."""
        self.current_resolution_area_idx += 1
        self.current_resolution_slot_idx = 0

    def reset_for_new_round(self) -> None:
        """Reset resolution tracking for a new round."""
        self.current_resolution_area_idx = 0
        self.current_resolution_slot_idx = 0
        self.round_number += 1


@dataclass
class GameState:
    """The complete game state - single source of truth.

    Combines all game components into a unified state that can be
    cloned, serialized, and hashed for RL algorithms.

    Attributes:
        board: The game board graph with nodes, edges, and occupancy.
        players: List of all players in turn order.
        action_board: The action board with marker placements.
        passenger_manager: Manager for all passengers.
        global_state: Global game state (round, time clock, etc.).
        phase: Current game phase.
    """

    board: BoardGraph
    players: list[Player]
    action_board: ActionBoard
    passenger_manager: PassengerManager
    global_state: GlobalState
    phase: Phase

    @classmethod
    def create_initial_state(
        cls,
        board: BoardGraph,
        num_players: int,
    ) -> GameState:
        """Create an initial game state with the given board and player count.

        Args:
            board: The game board graph (already loaded with topology).
            num_players: Number of players (3-5).

        Returns:
            A new GameState ready for the setup phase.

        Raises:
            ValueError: If num_players is out of valid range.
        """
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(
                f"Number of players must be between {MIN_PLAYERS} and {MAX_PLAYERS}, "
                f"got {num_players}"
            )

        # Create players
        players = [Player(player_id=i) for i in range(num_players)]

        # Create action board
        action_board = ActionBoard()

        # Create passenger manager
        passenger_manager = PassengerManager()

        # Create global state
        global_state = GlobalState()
        
        # Adjust time stones for player count
        # 3 players: 4 time stones
        # 4-5 players: 5 time stones
        if num_players == 3:
            global_state.time_stones_remaining = 4
        else:
            global_state.time_stones_remaining = 5

        return cls(
            board=board,
            players=players,
            action_board=action_board,
            passenger_manager=passenger_manager,
            global_state=global_state,
            phase=Phase.SETUP_BUILDINGS,
        )

    # -------------------------------------------------------------------------
    # Player access methods
    # -------------------------------------------------------------------------

    def get_current_player(self) -> Player:
        """Get the current player."""
        return self.players[self.global_state.current_player_idx]

    def get_player(self, player_id: int) -> Player:
        """Get a player by ID.

        Raises:
            ValueError: If player_id is invalid.
        """
        if not 0 <= player_id < len(self.players):
            raise ValueError(f"Invalid player ID: {player_id}")
        return self.players[player_id]

    def get_starting_player(self) -> Player:
        """Get the starting player for the current round."""
        return self.players[self.global_state.starting_player_idx]

    def num_players(self) -> int:
        """Return the number of players."""
        return len(self.players)

    def advance_current_player(self) -> None:
        """Move to the next player in turn order."""
        self.global_state.current_player_idx = (
            (self.global_state.current_player_idx + 1) % len(self.players)
        )

    def set_starting_player(self, player_id: int) -> None:
        """Set the starting player for the next round.

        Args:
            player_id: The player ID to become starting player.

        Raises:
            ValueError: If player_id is invalid.
        """
        if not 0 <= player_id < len(self.players):
            raise ValueError(f"Invalid player ID: {player_id}")
        self.global_state.starting_player_idx = player_id

    # -------------------------------------------------------------------------
    # Phase management
    # -------------------------------------------------------------------------

    def set_phase(self, phase: Phase) -> None:
        """Set the current game phase."""
        self.phase = phase

    def is_setup_phase(self) -> bool:
        """Check if the game is in a setup phase."""
        return self.phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        )

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.phase == Phase.GAME_OVER or self.global_state.game_ended

    # -------------------------------------------------------------------------
    # Round management
    # -------------------------------------------------------------------------

    def start_new_round(self) -> None:
        """Reset state for a new round."""
        # Reset player per-round state
        for player in self.players:
            player.reset_for_new_round()

        # Clear action board
        self.action_board.clear_all()

        # Reset global state resolution tracking
        self.global_state.reset_for_new_round()

        # Set current player to starting player
        self.global_state.current_player_idx = self.global_state.starting_player_idx

        # Set phase to choosing actions
        self.phase = Phase.CHOOSING_ACTIONS

    def all_players_passed(self) -> bool:
        """Check if all players have passed."""
        return all(player.has_passed for player in self.players)

    def get_active_players(self) -> list[Player]:
        """Get all players who have not passed."""
        return [player for player in self.players if not player.has_passed]

    # -------------------------------------------------------------------------
    # Passenger access
    # -------------------------------------------------------------------------

    def get_passengers_at_node(self, node_id: NodeId) -> list[int]:
        """Get passenger IDs at a specific node."""
        return [p.passenger_id for p in self.passenger_manager.get_passengers_at(node_id)]

    def get_total_passengers(self) -> int:
        """Get the total number of passengers in the game."""
        return self.passenger_manager.count()

    # -------------------------------------------------------------------------
    # Cloning and serialization
    # -------------------------------------------------------------------------

    def clone(self) -> GameState:
        """Create a deep copy of the game state.

        Used for RL rollouts and hypothetical state exploration.

        Returns:
            A complete deep copy of this GameState.
        """
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the game state to a dictionary.

        Useful for saving/loading games and debugging.

        Returns:
            Dictionary representation of the game state.
        """
        return {
            "phase": self.phase.value,
            "global_state": {
                "round_number": self.global_state.round_number,
                "current_player_idx": self.global_state.current_player_idx,
                "starting_player_idx": self.global_state.starting_player_idx,
                "time_clock_position": self.global_state.time_clock_position.value,
                "time_stones_remaining": self.global_state.time_stones_remaining,
                "current_resolution_area_idx": self.global_state.current_resolution_area_idx,
                "current_resolution_slot_idx": self.global_state.current_resolution_slot_idx,
                "game_ended": self.global_state.game_ended,
            },
            "players": [
                {
                    "player_id": p.player_id,
                    "action_markers_remaining": p.action_markers_remaining,
                    "rail_segments_remaining": p.rail_segments_remaining,
                    "buses": p.buses,
                    "score": p.score,
                    "time_stones": p.time_stones,
                    "has_passed": p.has_passed,
                    "markers_placed_this_round": p.markers_placed_this_round,
                }
                for p in self.players
            ],
            "action_board": {
                "placement_counter": self.action_board.placement_counter,
                "areas": {
                    area_type.value: {
                        "slots": {
                            label: {
                                "player_id": slot.player_id,
                                "placement_order": slot.placement_order,
                            }
                            for label, slot in area.slots.items()
                        }
                    }
                    for area_type, area in self.action_board.areas.items()
                },
            },
            "passengers": {
                str(p.passenger_id): p.location
                for p in self.passenger_manager.passengers.values()
            },
            "board_state": self._serialize_board_state(),
        }

    def _serialize_board_state(self) -> dict[str, Any]:
        """Serialize the mutable board state (buildings, rails, passengers).

        Note: This only serializes the mutable state, not the topology.
        """
        return {
            "nodes": {
                str(node_id): {
                    "building_slots": [
                        {
                            "zone": slot.zone.value,
                            "building": slot.building.value if slot.building else None,
                            "occupied_by_passenger_id": slot.occupied_by_passenger_id,
                        }
                        for slot in node.building_slots
                    ],
                    "passenger_ids": list(node.passenger_ids),
                }
                for node_id, node in self.board.nodes.items()
            },
            "edges": {
                f"{edge_id[0]},{edge_id[1]}": {
                    "rail_segments": [
                        {"player_id": seg.player_id}
                        for seg in edge.rail_segments
                    ]
                }
                for edge_id, edge in self.board.edges.items()
            },
        }

    def state_hash(self) -> str:
        """Compute a hash of the game state.

        Useful for detecting duplicate states in RL algorithms
        and for state caching.

        Returns:
            A hex string hash of the serialized state.
        """
        state_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the game state for consistency.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Check player count
        if not MIN_PLAYERS <= len(self.players) <= MAX_PLAYERS:
            errors.append(
                f"Invalid player count: {len(self.players)} "
                f"(must be {MIN_PLAYERS}-{MAX_PLAYERS})"
            )

        # Check player IDs are sequential
        for i, player in enumerate(self.players):
            if player.player_id != i:
                errors.append(
                    f"Player at index {i} has ID {player.player_id} (expected {i})"
                )

        # Check current player index is valid
        if not 0 <= self.global_state.current_player_idx < len(self.players):
            errors.append(
                f"Invalid current_player_idx: {self.global_state.current_player_idx}"
            )

        # Check starting player index is valid
        if not 0 <= self.global_state.starting_player_idx < len(self.players):
            errors.append(
                f"Invalid starting_player_idx: {self.global_state.starting_player_idx}"
            )

        # Check time stones are non-negative
        if self.global_state.time_stones_remaining < 0:
            errors.append(
                f"Negative time stones remaining: {self.global_state.time_stones_remaining}"
            )

        # Check time clock position is valid
        if self.global_state.time_clock_position not in TIME_CLOCK_ORDER:
            errors.append(
                f"Invalid time clock position: {self.global_state.time_clock_position}"
            )

        # Check passenger locations match node passenger_ids
        for passenger in self.passenger_manager.passengers.values():
            node = self.board.nodes.get(passenger.location)
            if node is None:
                errors.append(
                    f"Passenger {passenger.passenger_id} at non-existent node {passenger.location}"
                )
            elif passenger.passenger_id not in node.passenger_ids:
                errors.append(
                    f"Passenger {passenger.passenger_id} location {passenger.location} "
                    f"not reflected in node.passenger_ids"
                )

        return errors

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        lines = [
            f"GameState(phase={self.phase.value}, round={self.global_state.round_number})",
            f"  Time clock: {self.global_state.time_clock_position.value}",
            f"  Time stones remaining: {self.global_state.time_stones_remaining}",
            f"  Current player: {self.global_state.current_player_idx}",
            f"  Starting player: {self.global_state.starting_player_idx}",
            f"  Players ({len(self.players)}):",
        ]
        for p in self.players:
            status = "PASSED" if p.has_passed else "active"
            lines.append(
                f"    P{p.player_id}: score={p.score}, buses={p.buses}, "
                f"markers={p.action_markers_remaining}, rails={p.rail_segments_remaining}, "
                f"[{status}]"
            )
        lines.append(f"  Passengers: {self.passenger_manager.count()}")
        lines.append(f"  Markers on action board: {self.action_board.count_total_markers()}")
        return "\n".join(lines)
