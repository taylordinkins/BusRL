"""Main game engine for the Bus board game.

The GameEngine is the primary interface for playing the game. It provides:
- reset(): Initialize a new game
- step(): Execute an action and advance game state
- get_valid_actions(): Return legal actions for the current state

The engine enforces all game rules and manages phase transitions.
Action legality is enforced, not learned - illegal actions are never executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING

from core.constants import (
    Phase,
    ActionAreaType,
    BuildingType,
    Zone,
    ACTION_RESOLUTION_ORDER,
    MIN_MARKERS_PER_ROUND,
    ZONE_PRIORITY,
)
from core.board import BoardGraph, NodeId, EdgeId, make_edge_id
from core.game_state import GameState
from core.player import Player
from data.loader import load_default_board

from .phase_machine import PhaseMachine
from .setup import SetupManager, initialize_game


class ActionType(Enum):
    """Types of actions a player can take."""

    # Choosing Actions phase
    PLACE_MARKER = "place_marker"
    PASS = "pass"

    # Setup phases
    PLACE_BUILDING_SETUP = "place_building_setup"
    PLACE_RAIL_SETUP = "place_rail_setup"

    # Resolution phase (placeholders - will be expanded with resolvers)
    RESOLVE_LINE_EXPANSION = "resolve_line_expansion"
    RESOLVE_BUSES = "resolve_buses"
    RESOLVE_PASSENGERS = "resolve_passengers"
    RESOLVE_BUILDINGS = "resolve_buildings"
    RESOLVE_TIME_CLOCK = "resolve_time_clock"
    RESOLVE_VRROOMM = "resolve_vrroomm"
    RESOLVE_STARTING_PLAYER = "resolve_starting_player"


@dataclass
class Action:
    """Represents an action to be executed.

    Attributes:
        action_type: The type of action.
        player_id: The player taking the action.
        params: Additional parameters for the action (context-dependent).
    """

    action_type: ActionType
    player_id: int
    params: dict[str, Any]

    def __str__(self) -> str:
        return f"Action({self.action_type.value}, player={self.player_id}, params={self.params})"


@dataclass
class StepResult:
    """Result of executing a step in the game.

    Attributes:
        success: Whether the action was executed successfully.
        state: The game state after the action.
        reward: Reward for the action (for RL, per player).
        done: Whether the game has ended.
        info: Additional information about the step.
    """

    success: bool
    state: GameState
    reward: dict[int, float]
    done: bool
    info: dict[str, Any]


class GameEngine:
    """Main engine for playing the Bus board game.

    The engine manages the game state, enforces rules, and provides
    the interface for both human players and RL agents.

    Usage:
        engine = GameEngine()
        engine.reset(num_players=4)

        while not engine.is_game_over():
            actions = engine.get_valid_actions()
            action = select_action(actions)  # Player or agent selects
            result = engine.step(action)
    """

    def __init__(self):
        """Initialize the game engine."""
        self._state: Optional[GameState] = None
        self._phase_machine: Optional[PhaseMachine] = None
        self._setup_manager: Optional[SetupManager] = None
        self._board: Optional[BoardGraph] = None

    @property
    def state(self) -> GameState:
        """Get the current game state.

        Raises:
            RuntimeError: If the game has not been initialized.
        """
        if self._state is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self._state

    @property
    def phase(self) -> Phase:
        """Get the current game phase."""
        return self.state.phase

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self._phase_machine is not None and self._phase_machine.is_game_over()

    # -------------------------------------------------------------------------
    # Game Initialization
    # -------------------------------------------------------------------------

    def reset(
        self,
        num_players: int = 4,
        board: Optional[BoardGraph] = None,
        seed: Optional[int] = None,
    ) -> GameState:
        """Initialize a new game.

        Args:
            num_players: Number of players (3-5).
            board: Optional custom board. If None, uses the default board.

        Returns:
            The initial game state.
        """
        # Load board
        self._board = board if board is not None else load_default_board()

        # Create initial state
        self._state = GameState.create_initial_state(self._board, num_players)
        
        # Apply seeding for diversity if provided
        if seed is not None:
            import random
            rng = random.Random(seed)
            # Randomize starting player for training diversity
            start_player = rng.randint(0, num_players - 1)
            self._state.set_starting_player(start_player)
            # Also set current player to starting player (it might be redundant but safe)
            self._state.global_state.current_player_idx = start_player

        # Initialize phase machine
        self._phase_machine = PhaseMachine(initial_phase=Phase.SETUP_BUILDINGS)

        # Initialize setup manager and place initial passengers
        self._setup_manager = initialize_game(self._state)

        return self._state

    # -------------------------------------------------------------------------
    # Action Execution
    # -------------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """Execute an action and advance the game state.

        Args:
            action: The action to execute.

        Returns:
            StepResult with the outcome of the action.
        """
        # Validate action
        valid_actions = self.get_valid_actions()
        if not self._is_action_valid(action, valid_actions):
            return StepResult(
                success=False,
                state=self.state,
                reward={p.player_id: 0.0 for p in self.state.players},
                done=self.is_game_over(),
                info={"error": f"Invalid action: {action}"},
            )

        # Execute based on action type
        reward = {p.player_id: 0.0 for p in self.state.players}
        info: dict[str, Any] = {"action": str(action)}

        try:
            if action.action_type == ActionType.PLACE_MARKER:
                self._execute_place_marker(action)
            elif action.action_type == ActionType.PASS:
                self._execute_pass(action)
            elif action.action_type == ActionType.PLACE_BUILDING_SETUP:
                self._execute_place_building_setup(action)
            elif action.action_type == ActionType.PLACE_RAIL_SETUP:
                self._execute_place_rail_setup(action)
            # Resolution actions will be handled by resolvers (Phase 4)
            else:
                info["warning"] = f"Action type {action.action_type} not yet implemented"

            # Check for phase transitions
            self._check_phase_transition()

            info["phase"] = self.state.phase.value
            info["round"] = self.state.global_state.round_number

        except Exception as e:
            return StepResult(
                success=False,
                state=self.state,
                reward=reward,
                done=self.is_game_over(),
                info={"error": str(e)},
            )

        return StepResult(
            success=True,
            state=self.state,
            reward=reward,
            done=self.is_game_over(),
            info=info,
        )

    def _is_action_valid(self, action: Action, valid_actions: list[Action]) -> bool:
        """Check if an action is in the list of valid actions."""
        for valid in valid_actions:
            if (
                action.action_type == valid.action_type
                and action.player_id == valid.player_id
                and action.params == valid.params
            ):
                return True
        return False

    # -------------------------------------------------------------------------
    # Action Execution Helpers
    # -------------------------------------------------------------------------

    def _execute_place_marker(self, action: Action) -> None:
        """Execute a place marker action."""
        player = self.state.get_player(action.player_id)
        area_type = ActionAreaType(action.params["area_type"])

        # Place marker on action board
        self.state.action_board.place_marker(area_type, action.player_id)

        # Deduct from player's markers
        player.place_marker()

        # Advance to next player
        self._advance_to_next_active_player()

    def _execute_pass(self, action: Action) -> None:
        """Execute a pass action."""
        player = self.state.get_player(action.player_id)
        player.pass_turn()

        # Advance to next player
        self._advance_to_next_active_player()

    def _execute_place_building_setup(self, action: Action) -> None:
        """Execute a building placement during setup."""
        node_id = action.params["node_id"]
        slot_index = action.params["slot_index"]
        building_type = BuildingType(action.params["building_type"])

        self._setup_manager.place_building(
            action.player_id, node_id, slot_index, building_type
        )

        # Check if player finished their buildings, advance to next player
        if self._setup_manager.is_player_building_setup_complete(action.player_id):
            next_player = self._setup_manager.get_next_player_buildings()
            if next_player is not None:
                self.state.global_state.current_player_idx = next_player

    def _execute_place_rail_setup(self, action: Action) -> None:
        """Execute a rail placement during setup."""
        edge_id = tuple(action.params["edge_id"])

        if self.state.phase == Phase.SETUP_RAILS_FORWARD:
            self._setup_manager.place_rail_forward(action.player_id, edge_id)
            next_player = self._setup_manager.get_next_player_rails_forward()
            if next_player is not None:
                self.state.global_state.current_player_idx = next_player
        elif self.state.phase == Phase.SETUP_RAILS_REVERSE:
            self._setup_manager.place_rail_reverse(action.player_id, edge_id)
            next_player = self._setup_manager.get_next_player_rails_reverse()
            if next_player is not None:
                self.state.global_state.current_player_idx = next_player

    def _advance_to_next_active_player(self) -> None:
        """Advance to the next player who hasn't passed."""
        start_idx = self.state.global_state.current_player_idx
        num_players = len(self.state.players)

        for _ in range(num_players):
            self.state.advance_current_player()
            current = self.state.get_current_player()
            if not current.has_passed:
                return

        # All players have passed - stay on current (will trigger phase transition)

    # -------------------------------------------------------------------------
    # Phase Transition Logic
    # -------------------------------------------------------------------------

    def _check_phase_transition(self) -> None:
        """Check and execute any necessary phase transitions."""
        if self._phase_machine is None:
            return

        current_phase = self.state.phase

        # Setup phase transitions
        if current_phase == Phase.SETUP_BUILDINGS:
            if self._setup_manager.is_buildings_phase_complete():
                self._transition_to_phase(Phase.SETUP_RAILS_FORWARD)
                # Set first player for forward rails
                next_player = self._setup_manager.get_next_player_rails_forward()
                if next_player is not None:
                    self.state.global_state.current_player_idx = next_player

        elif current_phase == Phase.SETUP_RAILS_FORWARD:
            if self._setup_manager.is_rails_forward_phase_complete():
                self._transition_to_phase(Phase.SETUP_RAILS_REVERSE)
                # Set first player for reverse rails (reverse order)
                next_player = self._setup_manager.get_next_player_rails_reverse()
                if next_player is not None:
                    self.state.global_state.current_player_idx = next_player

        elif current_phase == Phase.SETUP_RAILS_REVERSE:
            if self._setup_manager.is_rails_reverse_phase_complete():
                self._transition_to_phase(Phase.CHOOSING_ACTIONS)
                # Set starting player
                self.state.global_state.current_player_idx = (
                    self.state.global_state.starting_player_idx
                )

        # Main game loop transitions
        elif current_phase == Phase.CHOOSING_ACTIONS:
            if self._phase_machine.should_end_choosing_phase(self.state):
                self._transition_to_phase(Phase.RESOLVING_ACTIONS)
                self.state.global_state.current_resolution_area_idx = 0
                self.state.global_state.current_resolution_slot_idx = 0

        elif current_phase == Phase.RESOLVING_ACTIONS:
            if self._phase_machine.should_end_resolving_phase(self.state):
                self._transition_to_phase(Phase.CLEANUP)
                self._execute_cleanup()

        elif current_phase == Phase.CLEANUP:
            # Automatic cleanup resolution
            self.resolve_cleanup()

    def _transition_to_phase(self, new_phase: Phase) -> None:
        """Transition to a new phase."""
        self._phase_machine.transition_to(new_phase)
        self.state.set_phase(new_phase)

    def _execute_cleanup(self) -> None:
        """Execute cleanup phase logic."""
        # Action markers are permanently removed (already tracked in player resources)
        # The action board is cleared in start_new_round()
        pass

    # -------------------------------------------------------------------------
    # Valid Actions
    # -------------------------------------------------------------------------

    def get_valid_actions(self) -> list[Action]:
        """Get all valid actions for the current state.

        Returns:
            List of valid actions the current player can take.
        """
        phase = self.state.phase

        if phase == Phase.SETUP_BUILDINGS:
            return self._get_valid_building_setup_actions()
        elif phase == Phase.SETUP_RAILS_FORWARD:
            return self._get_valid_rail_forward_setup_actions()
        elif phase == Phase.SETUP_RAILS_REVERSE:
            return self._get_valid_rail_reverse_setup_actions()
        elif phase == Phase.CHOOSING_ACTIONS:
            return self._get_valid_choosing_actions()
        elif phase == Phase.RESOLVING_ACTIONS:
            return self._get_valid_resolution_actions()
        elif phase == Phase.GAME_OVER:
            return []
        else:
            return []

    def _get_valid_building_setup_actions(self) -> list[Action]:
        """Get valid building placement actions during setup."""
        actions: list[Action] = []
        current_player_id = self.state.global_state.current_player_idx

        # Get valid slots from setup manager
        valid_slots = self._setup_manager.get_valid_building_slots()

        # Each valid slot can have any building type
        for node_id, slot_index in valid_slots:
            for building_type in BuildingType:
                actions.append(
                    Action(
                        action_type=ActionType.PLACE_BUILDING_SETUP,
                        player_id=current_player_id,
                        params={
                            "node_id": node_id,
                            "slot_index": slot_index,
                            "building_type": building_type.value,
                        },
                    )
                )

        return actions

    def _get_valid_rail_forward_setup_actions(self) -> list[Action]:
        """Get valid rail placement actions during forward setup."""
        actions: list[Action] = []
        current_player_id = self.state.global_state.current_player_idx

        valid_edges = self._setup_manager.get_valid_rail_edges_forward(current_player_id)

        for edge_id in valid_edges:
            actions.append(
                Action(
                    action_type=ActionType.PLACE_RAIL_SETUP,
                    player_id=current_player_id,
                    params={"edge_id": list(edge_id)},
                )
            )

        return actions

    def _get_valid_rail_reverse_setup_actions(self) -> list[Action]:
        """Get valid rail placement actions during reverse setup."""
        actions: list[Action] = []
        current_player_id = self.state.global_state.current_player_idx

        valid_edges = self._setup_manager.get_valid_rail_edges_reverse(current_player_id)

        for edge_id in valid_edges:
            actions.append(
                Action(
                    action_type=ActionType.PLACE_RAIL_SETUP,
                    player_id=current_player_id,
                    params={"edge_id": list(edge_id)},
                )
            )

        return actions

    def _get_valid_choosing_actions(self) -> list[Action]:
        """Get valid actions during choosing phase."""
        actions: list[Action] = []
        current_player = self.state.get_current_player()
        player_id = current_player.player_id

        # Cannot act if passed
        if current_player.has_passed:
            return []

        # Get available areas on action board
        available_areas = self.state.action_board.get_available_areas()

        # Can place marker if has markers remaining
        if current_player.can_place_marker():
            for area_type in available_areas:
                actions.append(
                    Action(
                        action_type=ActionType.PLACE_MARKER,
                        player_id=player_id,
                        params={"area_type": area_type.value},
                    )
                )

        # Can pass if:
        # - Has placed at least MIN_MARKERS_PER_ROUND markers, OR
        # - Has no markers remaining
        can_pass = (
            current_player.markers_placed_this_round >= MIN_MARKERS_PER_ROUND
            or current_player.action_markers_remaining == 0
        )

        if can_pass:
            actions.append(
                Action(
                    action_type=ActionType.PASS,
                    player_id=player_id,
                    params={},
                )
            )

        return actions

    def _get_valid_resolution_actions(self) -> list[Action]:
        """Get valid actions during resolution phase.

        Converts resolver-format actions to Action objects for the RL environment.
        The GUI uses ActionResolver directly, so this method is primarily for RL.

        For RL training, we return a NOOP action that triggers auto-advance
        through the resolution phase rather than exposing individual resolution choices.
        """
        # For RL, return NOOP to trigger auto-resolution
        # The environment's _auto_resolve_actions() handles the actual resolution
        current_player_id = self.state.global_state.current_player_idx
        return [
            Action(
                action_type=ActionType.PASS,
                player_id=current_player_id,
                params={"noop": True, "auto_resolve": True},
            )
        ]

    # -------------------------------------------------------------------------
    # Resolution Phase Support (for RL integration)
    # -------------------------------------------------------------------------

    def get_or_create_action_resolver(self) -> "ActionResolver":
        """Get or create an ActionResolver for the current resolution phase.

        This method is primarily for RL integration. The GUI uses its own
        ActionResolver instance managed by GameController.

        Returns:
            ActionResolver instance for the current state.

        Note:
            The resolver is recreated each time to ensure fresh state.
            For step-by-step resolution, callers should maintain their
            own resolver reference.
        """
        from .action_resolver import ActionResolver
        return ActionResolver(self.state)

    def get_valid_resolution_actions_for_rl(self) -> list[dict]:
        """Get valid resolution actions in dict format for RL.

        This method provides resolution-phase actions in a format
        suitable for the RL action space mapping.

        Returns:
            List of action dictionaries, or empty list if not in
            resolution phase or no actions available.
        """
        if self.state.phase != Phase.RESOLVING_ACTIONS:
            return []

        from .action_resolver import ActionResolver, ResolutionStatus

        resolver = ActionResolver(self.state)
        resolver.start_resolution()

        context = resolver.get_context()
        if context.status == ResolutionStatus.AWAITING_INPUT:
            return context.valid_actions

        return []

    def resolve_cleanup(self) -> StepResult:
        """Perform cleanup actions and transition to the next round.
        
        This handles the CLEANUP phase logic:
        1. Checks for end game conditions
        2. If game continues, resets for new round
        3. Transitions phase
        """
        if self.state.phase != Phase.CLEANUP:
            return StepResult(success=False, info={"error": "Not in CLEANUP phase"})

        next_phase_result = self._phase_machine.compute_next_phase(self.state)
        
        if next_phase_result.success:
            new_phase = next_phase_result.new_phase
            
            # Use unified transition helper to keep machine in sync
            self._transition_to_phase(new_phase)
            
            if new_phase == Phase.CHOOSING_ACTIONS:
                # Start new round logic (clears board, resets passed status)
                self.state.start_new_round()
            elif new_phase == Phase.GAME_OVER:
                self.state.global_state.game_ended = True
                
            return StepResult(
                success=True, 
                state=self.state,
                reward={p.player_id: 0.0 for p in self.state.players},
                done=(new_phase == Phase.GAME_OVER),
                info={"new_phase": new_phase.value, "reason": next_phase_result.reason}
            )
        else:
            return StepResult(
                success=False, 
                state=self.state,
                reward={p.player_id: 0.0 for p in self.state.players},
                done=False,
                info={"error": f"Cleanup failed: {next_phase_result.reason}"}
            )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_current_player(self) -> Player:
        """Get the current player."""
        return self.state.get_current_player()

    def get_max_buses(self) -> int:
        """Get the Maximum Number of Buses (M#oB).

        This is the highest number of buses owned by any player.
        """
        return max(p.buses for p in self.state.players)

    def clone(self) -> GameEngine:
        """Create a deep copy of the engine for simulation.

        Returns:
            A new GameEngine with cloned state.
        """
        new_engine = GameEngine()
        new_engine._state = self.state.clone()
        new_engine._board = self._board  # Board topology is immutable
        new_engine._phase_machine = PhaseMachine(initial_phase=self.state.phase)

        # Recreate setup manager with cloned state
        if self._setup_manager is not None:
            new_engine._setup_manager = SetupManager(new_engine._state)
            # Copy setup progress
            new_engine._setup_manager._buildings_placed = dict(
                self._setup_manager._buildings_placed
            )
            new_engine._setup_manager._rails_placed_forward = dict(
                self._setup_manager._rails_placed_forward
            )
            new_engine._setup_manager._rails_placed_reverse = dict(
                self._setup_manager._rails_placed_reverse
            )

        return new_engine

    def get_game_summary(self) -> dict[str, Any]:
        """Get a summary of the current game state.

        Returns:
            Dictionary with game summary information.
        """
        return {
            "phase": self.state.phase.value,
            "round": self.state.global_state.round_number,
            "current_player": self.state.global_state.current_player_idx,
            "time_clock": self.state.global_state.time_clock_position.value,
            "time_stones_remaining": self.state.global_state.time_stones_remaining,
            "max_buses": self.get_max_buses(),
            "players": [
                {
                    "id": p.player_id,
                    "score": p.score,
                    "buses": p.buses,
                    "markers_remaining": p.action_markers_remaining,
                    "rails_remaining": p.rail_segments_remaining,
                    "has_passed": p.has_passed,
                }
                for p in self.state.players
            ],
            "markers_on_board": self.state.action_board.count_total_markers(),
            "total_passengers": self.state.passenger_manager.count(),
            "game_over": self.is_game_over(),
        }

    def __str__(self) -> str:
        """Return string representation of the engine."""
        if self._state is None:
            return "GameEngine(not initialized)"
        return f"GameEngine(phase={self.state.phase.value}, round={self.state.global_state.round_number})"
