"""GUI renderer implementing the GameRenderer interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.game_state import GameState
from engine.driver import GameRenderer

if TYPE_CHECKING:
    from gui.main_window import MainWindow


class GUIRenderer(GameRenderer):
    """GUI implementation of GameRenderer.

    This class adapts the abstract GameRenderer interface to the GUI widgets.
    It delegates rendering to the MainWindow and its child widgets.
    """

    def __init__(self, main_window: MainWindow):
        """Initialize the GUI renderer.

        Args:
            main_window: The main application window.
        """
        self._window = main_window

    def render_state(self, state: GameState) -> None:
        """Render the full game state."""
        self._window.set_state(state)

    def render_phase_header(self, state: GameState) -> None:
        """Render the current phase header.

        This is handled by set_state() updating GameInfoWidget.
        """
        pass

    def render_player_status(self, state: GameState) -> None:
        """Render all players' status.

        This is handled by set_state() updating PlayerInfoWidget.
        """
        pass

    def render_action_board(self, state: GameState) -> None:
        """Render the action board.

        This is handled by set_state() updating ActionBoardWidget.
        """
        pass

    def render_message(self, message: str) -> None:
        """Render a message to the user."""
        self._window.add_message(message)

    def render_error(self, error: str) -> None:
        """Render an error message."""
        self._window.add_error(error)

    def render_game_over(self, state: GameState) -> None:
        """Render the game over screen."""
        self._window.show_game_over(state)

    def render_board_graph(self, state: GameState) -> None:
        """Render the board graph visualization.

        This is handled by set_state() updating BoardWidget.
        """
        pass
