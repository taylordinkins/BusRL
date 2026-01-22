"""Player model for the Bus game engine.

Each player has resources (action markers, rail segments, buses) and a score.
Resources are consumed during the game and are not recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
    MAX_BUSES,
    INITIAL_BUSES,
)


@dataclass
class Player:
    """Represents a player in the Bus game.

    Attributes:
        player_id: Unique identifier for this player (0-indexed).
        action_markers_remaining: Number of action markers not yet used.
        rail_segments_remaining: Number of rail segments not yet placed.
        buses: Current number of buses owned.
        score: Current victory points.
        time_stones: Number of time stones taken (each worth -1 point at game end).
        has_passed: Whether the player has passed this round.
        markers_placed_this_round: Number of markers placed in current round.
    """

    player_id: int
    action_markers_remaining: int = TOTAL_ACTION_MARKERS
    rail_segments_remaining: int = TOTAL_RAIL_SEGMENTS
    buses: int = INITIAL_BUSES
    score: int = 0
    time_stones: int = 0
    has_passed: bool = False
    markers_placed_this_round: int = 0
    network_endpoints: set[int] = field(default_factory=set)

    def can_place_marker(self) -> bool:
        """Check if the player can place an action marker."""
        return self.action_markers_remaining > 0 and not self.has_passed

    def place_marker(self) -> None:
        """Use one action marker.

        Raises:
            ValueError: If no markers remaining or player has passed.
        """
        if not self.can_place_marker():
            if self.has_passed:
                raise ValueError(f"Player {self.player_id} has already passed")
            raise ValueError(f"Player {self.player_id} has no action markers remaining")
        self.action_markers_remaining -= 1
        self.markers_placed_this_round += 1

    def can_place_rail(self) -> bool:
        """Check if the player can place a rail segment."""
        return self.rail_segments_remaining > 0

    def place_rail(self) -> None:
        """Use one rail segment.

        Raises:
            ValueError: If no rail segments remaining.
        """
        if not self.can_place_rail():
            raise ValueError(f"Player {self.player_id} has no rail segments remaining")
        self.rail_segments_remaining -= 1

    def can_gain_bus(self) -> bool:
        """Check if the player can gain another bus."""
        return self.buses < MAX_BUSES

    def gain_bus(self) -> None:
        """Gain one bus.

        Raises:
            ValueError: If already at maximum buses.
        """
        if not self.can_gain_bus():
            raise ValueError(f"Player {self.player_id} already has maximum buses ({MAX_BUSES})")
        self.buses += 1

    def add_score(self, points: int) -> None:
        """Add points to the player's score.

        Args:
            points: Number of points to add (can be negative).
        """
        self.score += points

    def take_time_stone(self) -> None:
        """Take a time stone (worth -1 point at game end)."""
        self.time_stones += 1

    def pass_turn(self) -> None:
        """Mark the player as having passed for this round.

        Raises:
            ValueError: If player has already passed.
        """
        if self.has_passed:
            raise ValueError(f"Player {self.player_id} has already passed")
        self.has_passed = True

    def reset_for_new_round(self) -> None:
        """Reset per-round state at the start of a new round."""
        self.has_passed = False
        self.markers_placed_this_round = 0

    def get_final_score(self) -> int:
        """Calculate final score including time stone penalty."""
        return self.score - self.time_stones

    def has_resources(self) -> bool:
        """Check if the player has any action markers remaining."""
        return self.action_markers_remaining > 0
