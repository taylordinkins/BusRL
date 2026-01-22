"""Interactive CLI driver for playing the Bus board game.

This module provides a text-based interface for playing Bus.
It serves as both a playable game and a reference implementation
for how a GUI would interact with the game engine.

The driver is designed to be extensible:
- GameRenderer handles all display logic (can be swapped for GUI)
- ActionPrompter handles all user input (can be swapped for GUI events)
- GameDriver orchestrates the game loop

Usage:
    python -m engine.driver

Or from code:
    from engine.driver import GameDriver
    driver = GameDriver(num_players=4)
    driver.run()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Callable, TYPE_CHECKING

from core.constants import (
    Phase,
    ActionAreaType,
    BuildingType,
    Zone,
    ACTION_RESOLUTION_ORDER,
)
from core.board import BoardGraph, NodeId, EdgeId
from core.game_state import GameState
from core.player import Player

from engine.game_engine import GameEngine, Action, ActionType, StepResult
from engine.action_resolver import (
    ActionResolver,
    ResolutionStatus,
    ResolutionContext,
)
from engine.resolvers import (
    TimeClockAction,
    PassengerDistribution,
    BuildingPlacement,
    RailPlacement,
)
from engine.resolvers.vrroomm import PassengerDelivery

# Optional visualization import
try:
    from data.graph_vis import visualize_board
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

if TYPE_CHECKING:
    from core.action_board import ActionSlot


# =============================================================================
# Display Formatters (GUI-ready abstraction)
# =============================================================================

class GameRenderer(ABC):
    """Abstract base class for rendering game state.

    Implement this interface to create a GUI renderer.
    The CLI renderer is provided as TextRenderer.
    """

    @abstractmethod
    def render_state(self, state: GameState) -> None:
        """Render the full game state."""
        pass

    @abstractmethod
    def render_phase_header(self, state: GameState) -> None:
        """Render the current phase header."""
        pass

    @abstractmethod
    def render_player_status(self, state: GameState) -> None:
        """Render all players' status."""
        pass

    @abstractmethod
    def render_action_board(self, state: GameState) -> None:
        """Render the action board."""
        pass

    @abstractmethod
    def render_message(self, message: str) -> None:
        """Render a message to the user."""
        pass

    @abstractmethod
    def render_error(self, error: str) -> None:
        """Render an error message."""
        pass

    @abstractmethod
    def render_game_over(self, state: GameState) -> None:
        """Render the game over screen."""
        pass

    @abstractmethod
    def render_board_graph(self, state: GameState) -> None:
        """Render the board graph visualization."""
        pass


class TextRenderer(GameRenderer):
    """CLI text-based renderer for the game state."""

    # Box drawing characters
    H_LINE = "─"
    V_LINE = "│"
    TL_CORNER = "┌"
    TR_CORNER = "┐"
    BL_CORNER = "└"
    BR_CORNER = "┘"
    T_DOWN = "┬"
    T_UP = "┴"
    T_RIGHT = "├"
    T_LEFT = "┤"
    CROSS = "┼"

    # Player colors (ANSI codes)
    PLAYER_COLORS = [
        "\033[91m",  # Red
        "\033[94m",  # Blue
        "\033[96m",  # Cyan
        "\033[93m",  # Yellow
        "\033[95m",  # Magenta
    ]
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, use_colors: bool = True):
        """Initialize the renderer.

        Args:
            use_colors: Whether to use ANSI color codes.
        """
        self.use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def _player_color(self, player_id: int) -> str:
        """Get the color code for a player."""
        return self.PLAYER_COLORS[player_id % len(self.PLAYER_COLORS)]

    def _box(self, title: str, content: list[str], width: int = 60) -> str:
        """Create a box around content."""
        lines = []
        # Top border with title
        title_space = width - len(title) - 4
        lines.append(f"{self.TL_CORNER}{self.H_LINE}{self.H_LINE} {title} {self.H_LINE * title_space}{self.TR_CORNER}")

        # Content
        for line in content:
            # Strip ANSI codes for padding calculation
            visible_len = len(self._strip_ansi(line))
            padding = width - visible_len - 2
            if padding < 0:
                padding = 0
            lines.append(f"{self.V_LINE} {line}{' ' * padding}{self.V_LINE}")

        # Bottom border
        lines.append(f"{self.BL_CORNER}{self.H_LINE * (width)}{self.BR_CORNER}")

        return "\n".join(lines)

    def _strip_ansi(self, text: str) -> str:
        """Strip ANSI escape codes from text."""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)

    def render_state(self, state: GameState) -> None:
        """Render the full game state."""
        print("\n" + "=" * 70)
        self.render_phase_header(state)
        print()
        self.render_player_status(state)
        print()
        self.render_board_summary(state)
        print()
        if state.phase == Phase.CHOOSING_ACTIONS:
            self.render_action_board(state)
            print()
        # Show board graph visualization
        self.render_board_graph(state)

    def render_phase_header(self, state: GameState) -> None:
        """Render the current phase header."""
        phase_names = {
            Phase.SETUP_BUILDINGS: "SETUP - Building Placement",
            Phase.SETUP_RAILS_FORWARD: "SETUP - Rail Placement (Forward)",
            Phase.SETUP_RAILS_REVERSE: "SETUP - Rail Placement (Reverse)",
            Phase.CHOOSING_ACTIONS: "CHOOSING ACTIONS",
            Phase.RESOLVING_ACTIONS: "RESOLVING ACTIONS",
            Phase.CLEANUP: "CLEANUP",
            Phase.GAME_OVER: "GAME OVER",
        }

        phase_name = phase_names.get(state.phase, state.phase.value)
        current_player = state.get_current_player()

        header = f"Round {state.global_state.round_number} - {phase_name}"
        player_str = self._color(
            f"Player {current_player.player_id}",
            self._player_color(current_player.player_id)
        )

        print(self._color(f"{'=' * 70}", self.DIM))
        print(self._color(header.center(70), self.BOLD))
        print(f"Current Turn: {player_str}")
        print(self._color(f"{'=' * 70}", self.DIM))

    def render_player_status(self, state: GameState) -> None:
        """Render all players' status."""
        content = []
        for player in state.players:
            color = self._player_color(player.player_id)
            current = " <--" if player.player_id == state.global_state.current_player_idx else ""
            passed = " [PASSED]" if player.has_passed else ""

            line = self._color(f"P{player.player_id}", color)
            line += f": Score={player.score}, Buses={player.buses}, "
            line += f"Markers={player.action_markers_remaining}, "
            line += f"Rails={player.rail_segments_remaining}"
            if player.time_stones > 0:
                line += f", TimeStones={player.time_stones}"
            line += self._color(passed, self.DIM) + current
            content.append(line)

        print(self._box("Players", content))

    def render_board_summary(self, state: GameState) -> None:
        """Render a summary of the board state."""
        content = []

        # Time clock
        clock = state.global_state.time_clock_position.value.upper()
        stones = state.global_state.time_stones_remaining
        content.append(f"Time Clock: {clock} | Time Stones: {stones}")
        content.append("")

        # Passenger count
        total_passengers = state.passenger_manager.count()
        content.append(f"Total Passengers: {total_passengers}")

        # Buildings placed
        buildings_placed = 0
        empty_slots = 0
        for node in state.board.nodes.values():
            for slot in node.building_slots:
                if slot.building:
                    buildings_placed += 1
                else:
                    empty_slots += 1
        content.append(f"Buildings Placed: {buildings_placed} | Empty Slots: {empty_slots}")

        # Rail segments
        rails_on_board = sum(
            len(edge.rail_segments) for edge in state.board.edges.values()
        )
        content.append(f"Rail Segments on Board: {rails_on_board}")

        # Max buses (M#oB)
        max_buses = max(p.buses for p in state.players)
        content.append(f"Maximum Number of Buses (M#oB): {max_buses}")

        print(self._box("Board Summary", content))

    def render_action_board(self, state: GameState) -> None:
        """Render the action board with placed markers."""
        content = []

        for area_type in ACTION_RESOLUTION_ORDER:
            area = state.action_board.get_area(area_type)
            slots = area.get_slots_in_resolution_order()

            # Build slot display
            slot_strs = []
            for slot in slots:
                if slot.is_empty():
                    slot_strs.append(f"[{slot.label}:__]")
                else:
                    p_color = self._player_color(slot.player_id)
                    player_marker = self._color(f"P{slot.player_id}", p_color)
                    slot_strs.append(f"[{slot.label}:{player_marker}]")

            area_name = area_type.value.replace("_", " ").title()
            content.append(f"{area_name:20} {' '.join(slot_strs)}")

        print(self._box("Action Board", content, width=70))

    def render_message(self, message: str) -> None:
        """Render a message to the user."""
        print(f"\n{message}")

    def render_error(self, error: str) -> None:
        """Render an error message."""
        print(self._color(f"\n[ERROR] {error}", "\033[91m"))

    def render_game_over(self, state: GameState) -> None:
        """Render the game over screen."""
        print("\n" + "=" * 70)
        print(self._color("GAME OVER".center(70), self.BOLD))
        print("=" * 70)

        # Sort players by score
        sorted_players = sorted(state.players, key=lambda p: p.score, reverse=True)

        print("\n" + self._color("FINAL SCORES", self.BOLD))
        print("-" * 40)
        for rank, player in enumerate(sorted_players, 1):
            color = self._player_color(player.player_id)
            stones_penalty = f" (-{player.time_stones} time stones)" if player.time_stones > 0 else ""
            final_score = player.score - player.time_stones
            print(f"  {rank}. {self._color(f'Player {player.player_id}', color)}: "
                  f"{final_score} points{stones_penalty}")

        winner = sorted_players[0]
        print(f"\n{self._color(f'Player {winner.player_id} WINS!', self.BOLD)}")
        print("=" * 70)

    def render_board_graph(self, state: GameState) -> None:
        """Render the board graph visualization using matplotlib."""
        if HAS_VISUALIZATION:
            visualize_board(
                state.board,
                title=f"Bus Game - Round {state.global_state.round_number} ({state.phase.value})",
                show=True,
                save_path=None,
            )
        else:
            print("[Board visualization not available - matplotlib not installed]")


# =============================================================================
# Action Prompter (GUI-ready abstraction)
# =============================================================================

@dataclass
class ActionChoice:
    """Represents a choice the player can make."""
    index: int
    description: str
    action: Any  # The action data to execute


class ActionPrompter(ABC):
    """Abstract base class for prompting player actions.

    Implement this interface to create a GUI action selector.
    The CLI prompter is provided as TextPrompter.
    """

    @abstractmethod
    def prompt_choice(
        self,
        message: str,
        choices: list[ActionChoice],
        allow_cancel: bool = False,
    ) -> Optional[ActionChoice]:
        """Prompt the player to make a choice.

        Args:
            message: The prompt message.
            choices: List of available choices.
            allow_cancel: Whether to allow canceling the choice.

        Returns:
            The selected choice, or None if canceled.
        """
        pass

    @abstractmethod
    def prompt_confirmation(self, message: str) -> bool:
        """Prompt for yes/no confirmation.

        Args:
            message: The confirmation message.

        Returns:
            True if confirmed, False otherwise.
        """
        pass


class TextPrompter(ActionPrompter):
    """CLI text-based action prompter."""

    def prompt_choice(
        self,
        message: str,
        choices: list[ActionChoice],
        allow_cancel: bool = False,
    ) -> Optional[ActionChoice]:
        """Prompt the player to make a choice."""
        print(f"\n{message}")
        print("-" * 50)

        for choice in choices:
            print(f"  {choice.index}. {choice.description}")

        if allow_cancel:
            print(f"  c. Cancel")

        print()
        while True:
            try:
                raw = input("Enter choice: ").strip().lower()

                if allow_cancel and raw == "c":
                    return None

                idx = int(raw)
                for choice in choices:
                    if choice.index == idx:
                        return choice

                print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return None

    def prompt_confirmation(self, message: str) -> bool:
        """Prompt for yes/no confirmation."""
        while True:
            raw = input(f"{message} (y/n): ").strip().lower()
            if raw in ("y", "yes"):
                return True
            elif raw in ("n", "no"):
                return False
            print("Please enter 'y' or 'n'.")


# =============================================================================
# Game Driver
# =============================================================================

class GameDriver:
    """Main driver for running an interactive game session.

    This class orchestrates the game loop and delegates to:
    - GameEngine for game logic
    - GameRenderer for display
    - ActionPrompter for user input

    To create a GUI version, simply provide different renderer and prompter.
    """

    def __init__(
        self,
        num_players: int = 4,
        renderer: Optional[GameRenderer] = None,
        prompter: Optional[ActionPrompter] = None,
    ):
        """Initialize the game driver.

        Args:
            num_players: Number of players (3-5).
            renderer: The renderer to use (default: TextRenderer).
            prompter: The prompter to use (default: TextPrompter).
        """
        self.engine = GameEngine()
        self.renderer = renderer or TextRenderer()
        self.prompter = prompter or TextPrompter()
        self.num_players = num_players
        self._action_resolver: Optional[ActionResolver] = None

    def run(self) -> None:
        """Run the main game loop."""
        # Initialize game
        self.engine.reset(num_players=self.num_players)
        self.renderer.render_message(f"Starting Bus game with {self.num_players} players!")

        # Main game loop
        while not self.engine.is_game_over():
            self.renderer.render_state(self.engine.state)

            phase = self.engine.state.phase

            if phase == Phase.SETUP_BUILDINGS:
                self._handle_setup_buildings()
            elif phase == Phase.SETUP_RAILS_FORWARD:
                self._handle_setup_rails_forward()
            elif phase == Phase.SETUP_RAILS_REVERSE:
                self._handle_setup_rails_reverse()
            elif phase == Phase.CHOOSING_ACTIONS:
                self._handle_choosing_actions()
            elif phase == Phase.RESOLVING_ACTIONS:
                self._handle_resolving_actions()
            elif phase == Phase.CLEANUP:
                # Execute cleanup logic
                self.engine.resolve_cleanup()
            elif phase == Phase.GAME_OVER:
                break

        # Game over
        self.renderer.render_game_over(self.engine.state)

    # -------------------------------------------------------------------------
    # Setup Phase Handlers
    # -------------------------------------------------------------------------

    def _handle_setup_buildings(self) -> None:
        """Handle the setup buildings phase."""
        valid_actions = self.engine.get_valid_actions()
        if not valid_actions:
            return

        current_player = self.engine.get_current_player()

        # Group actions by node for better display
        node_actions: dict[int, list[Action]] = {}
        for action in valid_actions:
            node_id = action.params["node_id"]
            if node_id not in node_actions:
                node_actions[node_id] = []
            node_actions[node_id].append(action)

        choices = []
        idx = 0
        for node_id in sorted(node_actions.keys()):
            for action in node_actions[node_id]:
                slot_idx = action.params["slot_index"]
                building = action.params["building_type"]
                node = self.engine.state.board.get_node(node_id)
                zone = node.building_slots[slot_idx].zone.value
                desc = f"Node {node_id}, Slot {slot_idx} (Zone {zone}): Place {building.upper()}"
                choices.append(ActionChoice(idx, desc, action))
                idx += 1

        choice = self.prompter.prompt_choice(
            f"Player {current_player.player_id}: Choose where to place a building",
            choices
        )

        if choice:
            result = self.engine.step(choice.action)
            if not result.success:
                self.renderer.render_error(result.info.get("error", "Unknown error"))

    def _handle_setup_rails_forward(self) -> None:
        """Handle the forward rails setup phase."""
        self._handle_rail_placement("place your first rail segment")

    def _handle_setup_rails_reverse(self) -> None:
        """Handle the reverse rails setup phase."""
        self._handle_rail_placement("place your second rail segment (must extend from first)")

    def _handle_rail_placement(self, message: str) -> None:
        """Handle rail placement during setup."""
        valid_actions = self.engine.get_valid_actions()
        if not valid_actions:
            return

        current_player = self.engine.get_current_player()

        choices = []
        for idx, action in enumerate(valid_actions):
            edge_id = tuple(action.params["edge_id"])
            desc = f"Edge {edge_id[0]} -- {edge_id[1]}"
            choices.append(ActionChoice(idx, desc, action))

        choice = self.prompter.prompt_choice(
            f"Player {current_player.player_id}: {message}",
            choices
        )

        if choice:
            result = self.engine.step(choice.action)
            if not result.success:
                self.renderer.render_error(result.info.get("error", "Unknown error"))

    # -------------------------------------------------------------------------
    # Choosing Actions Phase Handler
    # -------------------------------------------------------------------------

    def _handle_choosing_actions(self) -> None:
        """Handle the choosing actions phase."""
        valid_actions = self.engine.get_valid_actions()
        if not valid_actions:
            return

        current_player = self.engine.get_current_player()

        choices = []
        idx = 0

        for action in valid_actions:
            if action.action_type == ActionType.PASS:
                choices.append(ActionChoice(idx, "PASS (end marker placement)", action))
            elif action.action_type == ActionType.PLACE_MARKER:
                area_name = action.params["area_type"].replace("_", " ").title()
                choices.append(ActionChoice(idx, f"Place marker in {area_name}", action))
            idx += 1

        # Sort to put PASS at the end
        choices.sort(key=lambda c: c.description == "PASS (end marker placement)")
        # Re-index after sort
        for i, choice in enumerate(choices):
            choice.index = i

        choice = self.prompter.prompt_choice(
            f"Player {current_player.player_id}: Choose an action (Markers remaining: {current_player.action_markers_remaining})",
            choices
        )

        if choice:
            result = self.engine.step(choice.action)
            if not result.success:
                self.renderer.render_error(result.info.get("error", "Unknown error"))

    # -------------------------------------------------------------------------
    # Resolution Phase Handler
    # -------------------------------------------------------------------------

    def _handle_resolving_actions(self) -> None:
        """Handle the resolution phase."""
        if self._action_resolver is None:
            self._action_resolver = ActionResolver(self.engine.state)
            self._action_resolver.start_resolution()

        while not self._action_resolver.is_complete():
            context = self._action_resolver.get_context()

            if context.status == ResolutionStatus.AWAITING_INPUT:
                self._handle_resolution_input(context)
            else:
                # Automatic resolution (Buses, Starting Player)
                self._action_resolver.advance()

            # Update display after each resolution step
            if not self._action_resolver.is_complete():
                self.renderer.render_state(self.engine.state)

        # Resolution complete - update state so phase machine knows we're done
        # Set current_resolution_area_idx to beyond the last area
        self.engine.state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)
        self._action_resolver = None
        # Trigger cleanup phase transition
        self.engine._check_phase_transition()

    def _handle_resolution_input(self, context: ResolutionContext) -> None:
        """Handle player input during resolution."""
        area_type = context.current_area

        if area_type == ActionAreaType.LINE_EXPANSION:
            self._handle_line_expansion_input(context)
        elif area_type == ActionAreaType.PASSENGERS:
            self._handle_passengers_input(context)
        elif area_type == ActionAreaType.BUILDINGS:
            self._handle_buildings_input(context)
        elif area_type == ActionAreaType.TIME_CLOCK:
            self._handle_time_clock_input(context)
        elif area_type == ActionAreaType.VRROOMM:
            self._handle_vrroomm_input(context)

    def _handle_line_expansion_input(self, context: ResolutionContext) -> None:
        """Handle line expansion player choice."""
        valid_actions = context.valid_actions
        if not valid_actions:
            self._action_resolver.advance()
            return

        player_id = valid_actions[0]["player_id"]

        choices = []
        for idx, action in enumerate(valid_actions):
            edge = action["edge_id"]
            from_node = action["from_endpoint"]
            desc = f"Extend from node {from_node} to edge {edge[0]}--{edge[1]}"
            choices.append(ActionChoice(idx, desc, action))

        self.renderer.render_message(f"LINE EXPANSION - Player {player_id}")
        choice = self.prompter.prompt_choice(
            "Choose where to extend your rail network:",
            choices
        )

        if choice:
            self._action_resolver.apply_action(choice.action)

    def _handle_passengers_input(self, context: ResolutionContext) -> None:
        """Handle passengers player choice."""
        valid_actions = context.valid_actions
        if not valid_actions:
            self._action_resolver.advance()
            return

        player_id = valid_actions[0]["player_id"]

        choices = []
        for idx, action in enumerate(valid_actions):
            dist = action["distribution"]
            station_strs = [f"Station {s}: {c}" for s, c in dist.items()]
            desc = ", ".join(station_strs)
            choices.append(ActionChoice(idx, desc, action))

        self.renderer.render_message(f"PASSENGERS - Player {player_id}")
        choice = self.prompter.prompt_choice(
            "Choose how to distribute passengers to train stations:",
            choices
        )

        if choice:
            self._action_resolver.apply_action(choice.action)

    def _handle_buildings_input(self, context: ResolutionContext) -> None:
        """Handle buildings player choice."""
        valid_actions = context.valid_actions
        if not valid_actions:
            self._action_resolver.advance()
            return

        player_id = valid_actions[0]["player_id"]

        choices = []
        for idx, action in enumerate(valid_actions):
            node_id = action["node_id"]
            slot_idx = action["slot_index"]
            building = action["building_type"]
            node = self.engine.state.board.get_node(node_id)
            zone = node.building_slots[slot_idx].zone.value
            desc = f"Node {node_id}, Slot {slot_idx} (Zone {zone}): {building.value.upper()}"
            choices.append(ActionChoice(idx, desc, action))

        self.renderer.render_message(f"BUILDINGS - Player {player_id}")
        choice = self.prompter.prompt_choice(
            "Choose where and what to build:",
            choices
        )

        if choice:
            self._action_resolver.apply_action(choice.action)

    def _handle_time_clock_input(self, context: ResolutionContext) -> None:
        """Handle time clock player choice."""
        valid_actions = context.valid_actions
        if not valid_actions:
            self._action_resolver.advance()
            return

        player_id = valid_actions[0]["player_id"]
        current_clock = self.engine.state.global_state.time_clock_position.value
        stones_left = self.engine.state.global_state.time_stones_remaining

        choices = []
        for idx, action in enumerate(valid_actions):
            action_type = action["action"]
            if action_type == TimeClockAction.ADVANCE_CLOCK or action_type.value == "advance_clock":
                desc = f"Advance time clock (currently on {current_clock.upper()})"
            else:
                desc = f"STOP clock and take time stone (-1 point, {stones_left} remaining)"
            choices.append(ActionChoice(idx, desc, action))

        self.renderer.render_message(f"TIME CLOCK - Player {player_id}")
        choice = self.prompter.prompt_choice(
            "Choose your time clock action:",
            choices
        )

        if choice:
            self._action_resolver.apply_action(choice.action)

    def _handle_vrroomm_input(self, context: ResolutionContext) -> None:
        """Handle Vrroomm! player choice."""
        valid_actions = context.valid_actions
        if not valid_actions:
            # No valid deliveries, skip
            self._action_resolver.skip_vrroomm_deliveries()
            return

        player_id = valid_actions[0]["player_id"]
        player = self.engine.state.get_player(player_id)
        resolver = self._action_resolver._vrroomm_resolver
        remaining = resolver.get_deliveries_remaining_for_current_slot()

        choices = []
        for idx, action in enumerate(valid_actions):
            p_id = action["passenger_id"]
            from_n = action["from_node"]
            to_n = action["to_node"]
            slot_idx = action["building_slot_index"]
            desc = f"Passenger {p_id}: Node {from_n} -> Node {to_n} (slot {slot_idx})"
            choices.append(ActionChoice(idx, desc, action))

        # Add skip option
        skip_idx = len(choices)
        choices.append(ActionChoice(skip_idx, "SKIP remaining deliveries", None))

        self.renderer.render_message(f"VRROOMM! - Player {player_id}")
        self.renderer.render_message(f"(Deliveries remaining: {remaining}, Buses: {player.buses})")
        choice = self.prompter.prompt_choice(
            "Choose a passenger to deliver:",
            choices
        )

        if choice and choice.action is not None:
            self._action_resolver.apply_action(choice.action)
        else:
            # Skip
            self._action_resolver.skip_vrroomm_deliveries()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for the CLI driver."""
    import sys

    print("=" * 60)
    print("BUS BOARD GAME - Interactive CLI".center(60))
    print("=" * 60)
    print()

    # Get number of players
    while True:
        try:
            raw = input("Enter number of players (3-5): ").strip()
            num_players = int(raw)
            if 3 <= num_players <= 5:
                break
            print("Please enter a number between 3 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

    # Run the game
    driver = GameDriver(num_players=num_players)
    try:
        driver.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
