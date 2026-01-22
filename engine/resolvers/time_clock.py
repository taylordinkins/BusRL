"""Time Clock action resolver for the Bus game engine.

This resolver handles the Time Clock action area. The player who placed
a marker here has a choice:
1. Advance the time clock (default) - move to the next building type
2. Stop the time clock - take a time stone (-1 point at game end)

If the last time stone is taken, the game ends immediately.

Time Clock order: House -> Office -> Pub -> House (clockwise)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core.constants import ActionAreaType, BuildingType

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


class TimeClockAction(Enum):
    """Possible actions for the Time Clock resolution."""

    ADVANCE_CLOCK = "advance_clock"
    STOP_CLOCK = "stop_clock"


@dataclass
class TimeClockResult:
    """Result of resolving the Time Clock action.

    Attributes:
        resolved: Whether the action was resolved.
        player_id: The player who resolved this action, or None if no marker.
        action_taken: The action that was taken (advance or stop).
        new_clock_position: The clock position after resolution.
        time_stone_taken: Whether a time stone was taken.
        time_stones_remaining: How many time stones remain after this action.
        game_ended: Whether taking the last time stone ended the game.
    """

    resolved: bool
    player_id: int | None = None
    action_taken: TimeClockAction | None = None
    new_clock_position: BuildingType | None = None
    time_stone_taken: bool = False
    time_stones_remaining: int | None = None
    game_ended: bool = False


class TimeClockResolver:
    """Resolves the Time Clock action.

    This resolver:
    1. Checks if a marker was placed in the Time Clock area
    2. If so, offers the player a choice:
       - Advance the clock (default, no penalty)
       - Stop the clock (take a time stone, -1 point at game end)
    3. If stop clock is chosen and it's the last time stone, the game ends

    This is a single-slot area but requires player choice during resolution.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state
        self._resolved = False

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Time Clock area.

        Returns:
            List of occupied slots (0 or 1 for this single-slot area).
        """
        return self.state.action_board.get_markers_to_resolve(ActionAreaType.TIME_CLOCK)

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area.
        
        For Time Clock, we always return True because the clock advances
        automatically if no markers are present.
        """
        # Always return True so ActionResolver visits us
        # If no markers, we will perform automatic advance in resolve()
        if self._resolved:
            return False
        return True

    def get_resolving_player_id(self) -> int | None:
        """Get the player ID who needs to resolve this action.

        Returns:
            Player ID if there's a marker, None otherwise.
        """
        markers = self.get_markers_to_resolve()
        if not markers:
            return None
        return markers[0].player_id

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Time Clock resolution.

        Returns:
            List of valid action dictionaries the player can choose from.
            Each dict contains:
            - action: TimeClockAction enum value
            - player_id: The player who must make the decision
        """
        if self._resolved:
            return []

        markers = self.get_markers_to_resolve()
        if not markers:
            return []

        player_id = markers[0].player_id
        actions = []

        # Can always advance the clock
        actions.append({
            "action": TimeClockAction.ADVANCE_CLOCK,
            "player_id": player_id,
        })

        # Can stop clock if time stones remain
        if self.state.global_state.time_stones_remaining > 0:
            actions.append({
                "action": TimeClockAction.STOP_CLOCK,
                "player_id": player_id,
            })

        return actions

    def can_stop_clock(self) -> bool:
        """Check if stopping the clock (taking a time stone) is possible."""
        return self.state.global_state.time_stones_remaining > 0

    def resolve(self, action: TimeClockAction = TimeClockAction.ADVANCE_CLOCK) -> TimeClockResult:
        """Resolve the Time Clock action with the given choice.

        Args:
            action: The action to take (advance or stop clock).
                   Defaults to advancing the clock.

        Returns:
            TimeClockResult with resolution details.

        Raises:
            ValueError: If trying to stop clock when no time stones remain.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            # No markers - Default Action: Advance Clock
            self.state.global_state.advance_time_clock()
            self._resolved = True
            return TimeClockResult(
                resolved=True,
                action_taken=TimeClockAction.ADVANCE_CLOCK,
                new_clock_position=self.state.global_state.time_clock_position,
                time_stones_remaining=self.state.global_state.time_stones_remaining
            )

        slot = markers[0]
        player_id = slot.player_id
        player = self.state.get_player(player_id)

        time_stone_taken = False
        game_ended = False

        if action == TimeClockAction.STOP_CLOCK:
            # Validate that stopping is allowed
            if self.state.global_state.time_stones_remaining <= 0:
                raise ValueError("Cannot stop clock: no time stones remaining")

            # Take time stone
            self.state.global_state.take_time_stone()
            player.take_time_stone()
            time_stone_taken = True

            # Check if this was the last time stone (game end condition)
            if self.state.global_state.time_stones_remaining == 0:
                game_ended = True
                self.state.global_state.game_ended = True

            # Clock position stays the same when stopped

        else:  # ADVANCE_CLOCK
            # Advance the time clock
            self.state.global_state.advance_time_clock()

        self._resolved = True

        return TimeClockResult(
            resolved=True,
            player_id=player_id,
            action_taken=action,
            new_clock_position=self.state.global_state.time_clock_position,
            time_stone_taken=time_stone_taken,
            time_stones_remaining=self.state.global_state.time_stones_remaining,
            game_ended=game_ended,
        )

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        For Time Clock, resolution is complete after resolve() is called
        with a valid action.
        """
        return self._resolved or not self.has_markers()
