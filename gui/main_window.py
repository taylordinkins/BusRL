"""Main window for the Bus GUI application."""

from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFrame, QLabel, QStatusBar, QMenuBar, QMenu, QMessageBox,
    QDialog, QSpinBox, QPushButton, QDialogButtonBox, QFormLayout,
    QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QAction, QCloseEvent

from core.game_state import GameState
from core.constants import Phase

from gui.widgets import BoardWidget, ActionBoardWidget, PlayerInfoWidget, GameInfoWidget
from gui.constants import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT


class MessageLogWidget(QFrame):
    """Widget for displaying game messages and action history."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("Game Log")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(title)

        # Text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas", 9))
        self._text.setMaximumHeight(120)
        layout.addWidget(self._text)

    def add_message(self, message: str, is_error: bool = False) -> None:
        """Add a message to the log."""
        if is_error:
            self._text.append(f'<span style="color: red;">{message}</span>')
        else:
            self._text.append(message)
        # Scroll to bottom
        scrollbar = self._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self) -> None:
        """Clear the message log."""
        self._text.clear()


class NewGameDialog(QDialog):
    """Dialog for starting a new game."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("New Game")
        self.setModal(True)
        self.setMinimumWidth(300)

        layout = QFormLayout(self)

        # Player count
        self._player_spin = QSpinBox()
        self._player_spin.setMinimum(3)
        self._player_spin.setMaximum(5)
        self._player_spin.setValue(4)
        layout.addRow("Number of Players:", self._player_spin)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_num_players(self) -> int:
        """Get the selected number of players."""
        return self._player_spin.value()


class MainWindow(QMainWindow):
    """Main application window for the Bus GUI.

    Signals:
        state_updated: Emitted when the game state is updated
        action_requested: Emitted when user performs an action
    """

    state_updated = Signal(object)  # GameState
    node_clicked = Signal(int)
    edge_clicked = Signal(tuple)
    building_slot_clicked = Signal(int, int)  # node_id, slot_index
    action_board_clicked = Signal(str, str)  # area_type, slot_label
    pass_clicked = Signal()  # Emitted when Pass button is clicked

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._state: Optional[GameState] = None

        self.setWindowTitle("Bus - Board Game")
        self.setMinimumSize(1000, 700)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_menu(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()

        # Game menu
        game_menu = menubar.addMenu("Game")

        new_action = QAction("New Game...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._on_new_game)
        game_menu.addAction(new_action)

        game_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        game_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_display)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self) -> None:
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top section: Game info and Player info (side-by-side)
        top_section = QFrame()
        top_section.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(8, 4, 8, 4)
        top_layout.setSpacing(20)

        self._game_info = GameInfoWidget()
        top_layout.addWidget(self._game_info)
        
        # Player info next to game info
        self._player_info = PlayerInfoWidget()
        top_layout.addWidget(self._player_info)
        
        top_layout.addStretch()

        main_layout.addWidget(top_section)

        # Middle section: Board and Action Board side by side
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Board
        board_frame = QFrame()
        board_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        board_layout = QVBoxLayout(board_frame)
        board_layout.setContentsMargins(4, 4, 4, 4)

        board_title = QLabel("Game Board")
        board_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        board_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        board_layout.addWidget(board_title)

        self._board_widget = BoardWidget()
        self._board_widget.node_clicked.connect(self._on_node_clicked)
        self._board_widget.edge_clicked.connect(self._on_edge_clicked)
        self._board_widget.building_slot_clicked.connect(self._on_building_slot_clicked)
        board_layout.addWidget(self._board_widget, stretch=1)

        splitter.addWidget(board_frame)

        # Right: Action Board
        action_frame = QFrame()
        action_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        action_frame.setMaximumWidth(450)
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(4, 4, 4, 4)
        action_layout.setSpacing(4)

        self._action_board_widget = ActionBoardWidget()
        self._action_board_widget.slot_clicked.connect(self._on_action_slot_clicked)
        self._action_board_widget.pass_clicked.connect(self._on_pass_clicked)
        action_layout.addWidget(self._action_board_widget)

        splitter.addWidget(action_frame)

        # Set initial sizes
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter, stretch=1)

        # Bottom section: Message log
        self._message_log = MessageLogWidget()
        main_layout.addWidget(self._message_log)

    def _setup_statusbar(self) -> None:
        """Setup the status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready - Start a new game with Game > New Game")

    def set_state(self, state: GameState) -> None:
        """Update the display with the current game state."""
        self._state = state

        self._board_widget.set_state(state)
        self._action_board_widget.set_state(state)
        self._player_info.set_state(state)
        self._game_info.set_state(state)

        # Update status bar
        current_player = state.get_current_player()
        phase_name = state.phase.value.replace("_", " ").title()
        self._statusbar.showMessage(
            f"Round {state.global_state.round_number} | "
            f"{phase_name} | "
            f"Player {current_player.player_id}'s turn"
        )

        self.state_updated.emit(state)

    def highlight_valid_nodes(self, nodes: set[int]) -> None:
        """Highlight valid node targets for the current action."""
        self._board_widget.set_highlighted_nodes(nodes)

    def highlight_valid_edges(self, edges: set[tuple]) -> None:
        """Highlight valid edge targets for the current action."""
        self._board_widget.set_highlighted_edges(edges)

    def highlight_valid_slots(self, slots: set[tuple[int, int]]) -> None:
        """Highlight valid building slot targets."""
        self._board_widget.set_highlighted_slots(slots)

    def highlight_valid_action_areas(self, areas: list) -> None:
        """Highlight valid action areas for marker placement."""
        self._action_board_widget.set_available_areas(areas)

    def set_distribution_preview(self, distribution: dict[int, int]) -> None:
        """Set distribution counts to preview on stations."""
        self._board_widget.set_distribution_preview(distribution)

    def set_pass_button_state(self, visible: bool, can_pass: bool = False, label: Optional[str] = None) -> None:
        """Show/hide and enable/disable the Pass button."""
        self._action_board_widget.set_pass_enabled(visible, can_pass, label)

    def clear_highlights(self) -> None:
        """Clear all highlights."""
        self._board_widget.clear_highlights()
        self._action_board_widget.clear_highlights()

    def add_message(self, message: str) -> None:
        """Add a message to the game log."""
        self._message_log.add_message(message)

    def add_error(self, error: str) -> None:
        """Add an error message to the game log."""
        self._message_log.add_message(error, is_error=True)

    def show_game_over(self, state: GameState) -> None:
        """Show the game over dialog."""
        from gui.dialogs import GameOverDialog
        dialog = GameOverDialog(state, self)
        dialog.exec()

    def _on_new_game(self) -> None:
        """Handle new game menu action."""
        dialog = NewGameDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            num_players = dialog.get_num_players()
            self._message_log.clear()
            self.add_message(f"Starting new game with {num_players} players...")
            # This will be connected to the game controller
            # For now, emit a signal or call a callback
            if hasattr(self, '_new_game_callback') and self._new_game_callback:
                self._new_game_callback(num_players)

    def set_new_game_callback(self, callback: Callable[[int], None]) -> None:
        """Set the callback for starting a new game."""
        self._new_game_callback = callback

    def _on_node_clicked(self, node_id: int) -> None:
        """Handle node click."""
        self.node_clicked.emit(node_id)

    def _on_edge_clicked(self, edge_id: tuple) -> None:
        """Handle edge click."""
        self.edge_clicked.emit(edge_id)

    def _on_building_slot_clicked(self, node_id: int, slot_index: int) -> None:
        """Handle building slot click."""
        self.building_slot_clicked.emit(node_id, slot_index)

    def _on_action_slot_clicked(self, area_type: str, slot_label: str) -> None:
        """Handle action board slot click."""
        self.action_board_clicked.emit(area_type, slot_label)

    def _on_pass_clicked(self) -> None:
        """Handle Pass button click."""
        self.pass_clicked.emit()

    def _refresh_display(self) -> None:
        """Refresh the display."""
        if self._state:
            self.set_state(self._state)

    def _show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Bus",
            "Bus - Digital Board Game\n\n"
            "A deterministic implementation of the board game Bus.\n\n"
            "Built with Python and PySide6."
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close."""
        reply = QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
