"""Phase state machine for the Bus game engine.

Manages phase transitions including:
- Setup phases (executed once at game start)
- Main game loop phases (repeated each round)
- End game detection

The phase machine enforces valid transitions and provides
the logic for when transitions should occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.constants import Phase, ActionAreaType, ACTION_RESOLUTION_ORDER

if TYPE_CHECKING:
    from core.game_state import GameState


# Valid phase transitions
PHASE_TRANSITIONS: dict[Phase, list[Phase]] = {
    # Setup phases (linear, once at game start)
    Phase.SETUP_BUILDINGS: [Phase.SETUP_RAILS_FORWARD],
    Phase.SETUP_RAILS_FORWARD: [Phase.SETUP_RAILS_REVERSE],
    Phase.SETUP_RAILS_REVERSE: [Phase.CHOOSING_ACTIONS],
    # Main game loop
    Phase.CHOOSING_ACTIONS: [Phase.RESOLVING_ACTIONS],
    Phase.RESOLVING_ACTIONS: [Phase.CLEANUP],
    Phase.CLEANUP: [Phase.CHOOSING_ACTIONS, Phase.GAME_OVER],
    # Terminal
    Phase.GAME_OVER: [],
}


@dataclass
class PhaseTransitionResult:
    """Result of a phase transition attempt.

    Attributes:
        success: Whether the transition was successful.
        new_phase: The new phase if successful, None otherwise.
        reason: Description of why the transition failed (if it did).
    """

    success: bool
    new_phase: Optional[Phase]
    reason: Optional[str] = None


class PhaseMachine:
    """State machine for managing game phase transitions.

    The phase machine tracks the current phase and enforces valid
    transitions based on game state conditions. It does not modify
    game state directly - it only computes what the next phase should be.

    Phases:
        Setup (executed once):
        - SETUP_BUILDINGS: First player places 2 buildings, then clockwise
        - SETUP_RAILS_FORWARD: Each player places 1 rail segment in order
        - SETUP_RAILS_REVERSE: Each player places 1 rail segment in reverse order

        Main loop (repeated each round):
        - CHOOSING_ACTIONS: Players place action markers
        - RESOLVING_ACTIONS: Actions are resolved in fixed order
        - CLEANUP: Markers removed, check end conditions
        - GAME_OVER: Terminal state
    """

    def __init__(self, initial_phase: Phase = Phase.SETUP_BUILDINGS):
        """Initialize the phase machine.

        Args:
            initial_phase: The starting phase (default: SETUP_BUILDINGS).
        """
        self._phase = initial_phase

    @property
    def phase(self) -> Phase:
        """Get the current phase."""
        return self._phase

    def get_valid_transitions(self) -> list[Phase]:
        """Get the list of valid next phases from the current phase.

        Returns:
            List of phases that can be transitioned to.
        """
        return PHASE_TRANSITIONS.get(self._phase, [])

    def can_transition_to(self, target_phase: Phase) -> bool:
        """Check if a transition to the target phase is valid.

        Args:
            target_phase: The phase to transition to.

        Returns:
            True if the transition is valid, False otherwise.
        """
        return target_phase in self.get_valid_transitions()

    def transition_to(self, target_phase: Phase) -> PhaseTransitionResult:
        """Attempt to transition to a new phase.

        Args:
            target_phase: The phase to transition to.

        Returns:
            PhaseTransitionResult indicating success or failure.
        """
        if not self.can_transition_to(target_phase):
            valid = self.get_valid_transitions()
            return PhaseTransitionResult(
                success=False,
                new_phase=None,
                reason=f"Cannot transition from {self._phase.value} to {target_phase.value}. "
                f"Valid transitions: {[p.value for p in valid]}",
            )

        self._phase = target_phase
        return PhaseTransitionResult(success=True, new_phase=target_phase)

    def is_setup_phase(self) -> bool:
        """Check if currently in a setup phase."""
        return self._phase in (
            Phase.SETUP_BUILDINGS,
            Phase.SETUP_RAILS_FORWARD,
            Phase.SETUP_RAILS_REVERSE,
        )

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self._phase == Phase.GAME_OVER

    def is_choosing_phase(self) -> bool:
        """Check if in the choosing actions phase."""
        return self._phase == Phase.CHOOSING_ACTIONS

    def is_resolving_phase(self) -> bool:
        """Check if in the resolving actions phase."""
        return self._phase == Phase.RESOLVING_ACTIONS

    # -------------------------------------------------------------------------
    # Phase transition logic helpers
    # -------------------------------------------------------------------------

    def should_end_choosing_phase(self, state: GameState) -> bool:
        """Check if the choosing actions phase should end.

        The choosing phase ends when all players have passed.

        Args:
            state: The current game state.

        Returns:
            True if choosing phase should end.
        """
        if self._phase != Phase.CHOOSING_ACTIONS:
            return False
        return state.all_players_passed()

    def should_end_resolving_phase(self, state: GameState) -> bool:
        """Check if the resolving actions phase should end.

        The resolving phase ends when all action areas have been resolved.

        Args:
            state: The current game state.

        Returns:
            True if resolving phase should end.
        """
        if self._phase != Phase.RESOLVING_ACTIONS:
            return False
        return state.global_state.current_resolution_area_idx >= len(ACTION_RESOLUTION_ORDER)

    def should_game_end(self, state: GameState) -> tuple[bool, Optional[str]]:
        """Check if any end game condition is met.

        End conditions:
        1. Last time stone is taken
        2. Only one player has action markers remaining
        3. All building locations are filled

        Args:
            state: The current game state.

        Returns:
            Tuple of (should_end, reason) where reason describes the condition met.
        """
        # Check time stones exhausted
        if state.global_state.time_stones_remaining == 0:
            return True, "All time stones have been taken"

        # Check if only one player has markers
        players_with_markers = sum(
            1 for p in state.players if p.action_markers_remaining > 0
        )
        if players_with_markers <= 1:
            return True, "Only one player has action markers remaining"

        # Check if all building slots are filled
        empty_slots = state.board.get_empty_slots()
        total_empty = sum(len(slots) for slots in empty_slots.values())
        if total_empty == 0:
            return True, "All building locations are filled"

        return False, None

    def compute_next_phase(self, state: GameState) -> PhaseTransitionResult:
        """Compute what the next phase should be based on game state.

        This is the main decision logic for phase transitions. It examines
        the current state and determines if a transition should occur.

        Args:
            state: The current game state.

        Returns:
            PhaseTransitionResult with the recommended next phase.
        """
        current = self._phase

        # Setup phase transitions (linear progression)
        if current == Phase.SETUP_BUILDINGS:
            # Transition when all setup buildings have been placed
            # (caller must ensure this condition is met)
            return PhaseTransitionResult(
                success=True, new_phase=Phase.SETUP_RAILS_FORWARD
            )

        if current == Phase.SETUP_RAILS_FORWARD:
            return PhaseTransitionResult(
                success=True, new_phase=Phase.SETUP_RAILS_REVERSE
            )

        if current == Phase.SETUP_RAILS_REVERSE:
            return PhaseTransitionResult(
                success=True, new_phase=Phase.CHOOSING_ACTIONS
            )

        # Main game loop transitions
        if current == Phase.CHOOSING_ACTIONS:
            if self.should_end_choosing_phase(state):
                return PhaseTransitionResult(
                    success=True, new_phase=Phase.RESOLVING_ACTIONS
                )
            return PhaseTransitionResult(
                success=False,
                new_phase=None,
                reason="Not all players have passed yet",
            )

        if current == Phase.RESOLVING_ACTIONS:
            if self.should_end_resolving_phase(state):
                return PhaseTransitionResult(success=True, new_phase=Phase.CLEANUP)
            return PhaseTransitionResult(
                success=False,
                new_phase=None,
                reason="Not all action areas have been resolved",
            )

        if current == Phase.CLEANUP:
            should_end, reason = self.should_game_end(state)
            if should_end:
                return PhaseTransitionResult(
                    success=True,
                    new_phase=Phase.GAME_OVER,
                    reason=reason,
                )
            return PhaseTransitionResult(
                success=True, new_phase=Phase.CHOOSING_ACTIONS
            )

        # Game over - no more transitions
        if current == Phase.GAME_OVER:
            return PhaseTransitionResult(
                success=False,
                new_phase=None,
                reason="Game has ended - no further transitions",
            )

        # Fallback (should not reach here)
        return PhaseTransitionResult(
            success=False,
            new_phase=None,
            reason=f"Unknown phase: {current}",
        )

    # -------------------------------------------------------------------------
    # Setup phase helpers
    # -------------------------------------------------------------------------

    def get_setup_buildings_count(self, num_players: int) -> dict[int, int]:
        """Get the number of buildings each player places during setup.

        During setup, the first player places 2 buildings, then each
        other player places 2 buildings clockwise.

        Args:
            num_players: Number of players in the game.

        Returns:
            Dict mapping player_id to number of buildings to place.
        """
        return {player_id: 2 for player_id in range(num_players)}

    def get_setup_rails_forward_order(self, num_players: int) -> list[int]:
        """Get player order for first rail segment placement.

        Players place in order: 0, 1, 2, ... (num_players - 1).

        Args:
            num_players: Number of players in the game.

        Returns:
            List of player IDs in placement order.
        """
        return list(range(num_players))

    def get_setup_rails_reverse_order(self, num_players: int) -> list[int]:
        """Get player order for second rail segment placement.

        Players place in reverse order: (num_players - 1), ..., 1, 0.

        Args:
            num_players: Number of players in the game.

        Returns:
            List of player IDs in placement order.
        """
        return list(range(num_players - 1, -1, -1))

    # -------------------------------------------------------------------------
    # Resolution tracking helpers
    # -------------------------------------------------------------------------

    def get_current_resolution_area(self, state: GameState) -> Optional[ActionAreaType]:
        """Get the action area currently being resolved.

        Args:
            state: The current game state.

        Returns:
            The ActionAreaType being resolved, or None if resolution is complete.
        """
        return state.global_state.get_current_resolution_area()

    def get_resolution_progress(self, state: GameState) -> tuple[int, int]:
        """Get the current resolution progress.

        Args:
            state: The current game state.

        Returns:
            Tuple of (current_area_index, total_areas).
        """
        return (
            state.global_state.current_resolution_area_idx,
            len(ACTION_RESOLUTION_ORDER),
        )

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """Return string representation of the phase machine."""
        return f"PhaseMachine(phase={self._phase.value})"

    def __repr__(self) -> str:
        """Return detailed representation of the phase machine."""
        return f"PhaseMachine(phase={self._phase!r})"
