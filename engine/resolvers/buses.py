"""Buses action resolver for the Bus game engine.

This resolver handles the Buses action area, which allows a player
to gain an additional bus. Gaining a bus immediately updates the
Maximum Number of Buses (M#oB) if this player now has the most buses.

Resolution: Increment the player's bus count by 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.constants import ActionAreaType, MAX_BUSES

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


@dataclass
class BusesResult:
    """Result of resolving the Buses action.

    Attributes:
        resolved: Whether the action was resolved (True if marker was present).
        player_id: The player who gained a bus, or None if no marker.
        new_bus_count: The player's new bus count after gaining a bus.
        new_max_buses: The new Maximum Number of Buses (M#oB) after this action.
        bus_gained: Whether a bus was actually gained (False if already at max).
    """

    resolved: bool
    player_id: int | None = None
    new_bus_count: int | None = None
    new_max_buses: int | None = None
    bus_gained: bool = False


class BusesResolver:
    """Resolves the Buses action.

    This resolver:
    1. Checks if a marker was placed in the Buses area
    2. If so, increases the player's bus count by 1 (up to MAX_BUSES)
    3. Updates M#oB if necessary

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
        """Get all markers in the Buses area.

        Returns:
            List of occupied slots (0 or 1 for this single-slot area).
        """
        return self.state.action_board.get_markers_to_resolve(ActionAreaType.BUSES)

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def get_max_buses(self) -> int:
        """Get the current Maximum Number of Buses (M#oB).

        M#oB is the highest number of buses owned by any player.
        """
        return max(p.buses for p in self.state.players)

    def resolve(self) -> BusesResult:
        """Resolve the Buses action.

        If a marker was placed, the player who placed it gains one bus
        (up to the maximum of MAX_BUSES).

        Returns:
            BusesResult with resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            return BusesResult(resolved=False)

        # There's only one slot, so at most one marker
        slot = markers[0]
        player_id = slot.player_id
        player = self.state.get_player(player_id)

        # Try to gain a bus
        bus_gained = False
        if player.can_gain_bus():
            player.gain_bus()
            bus_gained = True

        # Calculate new M#oB
        new_max_buses = self.get_max_buses()

        return BusesResult(
            resolved=True,
            player_id=player_id,
            new_bus_count=player.buses,
            new_max_buses=new_max_buses,
            bus_gained=bus_gained,
        )

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Buses resolution.

        Since this action is automatic (no player choice), returns empty list.
        The game engine handles this by auto-advancing when there are no choices.

        Returns:
            Empty list - no player choices for this action.
        """
        # Buses resolution requires no player decisions
        # It's fully determined by who placed the marker
        return []

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        For Buses, resolution is complete after a single call
        to resolve() since there's no player interaction needed.
        """
        # This area is single-slot and requires no player choice
        # Resolution is instantaneous
        return True
