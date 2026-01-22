"""Starting Player action resolver for the Bus game engine.

This resolver handles the Starting Player action area, which assigns
the starting player for the next round. The player who placed a marker
in this area becomes the starting player.

Resolution: Assign starting player for next round to the player who
placed the marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.constants import ActionAreaType

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


@dataclass
class StartingPlayerResult:
    """Result of resolving the Starting Player action.

    Attributes:
        resolved: Whether the action was resolved (True if marker was present).
        new_starting_player_id: The player who becomes starting player, or None if no marker.
    """

    resolved: bool
    new_starting_player_id: int | None = None


class StartingPlayerResolver:
    """Resolves the Starting Player action.

    This resolver:
    1. Checks if a marker was placed in the Starting Player area
    2. If so, assigns the starting player for the next round to that player

    Since this is a single-slot action area, there's no player choice involved.
    The action is fully deterministic once the marker is placed.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Starting Player area.

        Returns:
            List of occupied slots (0 or 1 for this single-slot area).
        """
        return self.state.action_board.get_markers_to_resolve(
            ActionAreaType.STARTING_PLAYER
        )

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def resolve(self) -> StartingPlayerResult:
        """Resolve the Starting Player action.

        If a marker was placed, the player who placed it becomes the
        starting player for the next round.

        Returns:
            StartingPlayerResult with resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            # No markers - Default Action: Rotate starting player clockwise
            current_starting = self.state.global_state.starting_player_idx
            num_players = len(self.state.players)
            next_player_id = (current_starting + 1) % num_players
            
            self.state.set_starting_player(next_player_id)
            
            return StartingPlayerResult(
                resolved=True,
                new_starting_player_id=next_player_id,
            )

        # There's only one slot, so at most one marker
        slot = markers[0]
        player_id = slot.player_id

        # Set the starting player for the next round
        self.state.set_starting_player(player_id)

        return StartingPlayerResult(
            resolved=True,
            new_starting_player_id=player_id,
        )

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Starting Player resolution.

        Since this action is automatic (no player choice), returns empty list.
        The game engine handles this by auto-advancing when there are no choices.

        Returns:
            Empty list - no player choices for this action.
        """
        # Starting Player resolution requires no player decisions
        # It's fully determined by who placed the marker
        return []

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        For Starting Player, resolution is complete after a single call
        to resolve() since there's no player interaction needed.
        """
        # This area is single-slot and requires no player choice
        # Resolution is instantaneous
        return True
