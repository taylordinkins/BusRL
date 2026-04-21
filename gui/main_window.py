"""Main window for the Bus GUI application."""

from __future__ import annotations

import os
from typing import Optional, Callable

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFrame, QLabel, QStatusBar, QMenuBar, QMenu, QMessageBox,
    QDialog, QSpinBox, QPushButton, QDialogButtonBox, QFormLayout,
    QTextEdit, QScrollArea, QRadioButton, QButtonGroup, QFileDialog,
    QGridLayout, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QAction, QCloseEvent

from core.game_state import GameState
from core.constants import Phase

from gui.widgets import BoardWidget, ActionBoardWidget, PlayerInfoWidget, GameInfoWidget
from gui.constants import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
from gui.setup_config import GameSetupConfig, PlayerConfig


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
    """Dialog for starting a new game with per-player Human / AI configuration."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("New Game")
        self.setModal(True)
        self.setMinimumWidth(480)

        self._player_rows: list[dict] = []  # per-player widget refs

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        # -- Player count --
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Number of Players:"))
        self._player_spin = QSpinBox()
        self._player_spin.setMinimum(3)
        self._player_spin.setMaximum(5)
        self._player_spin.setValue(4)
        count_row.addWidget(self._player_spin)
        count_row.addStretch()
        outer.addLayout(count_row)

        # -- Per-player configuration area --
        self._players_box = QGroupBox("Player Configuration")
        self._players_layout = QGridLayout(self._players_box)
        self._players_layout.setColumnStretch(2, 1)
        outer.addWidget(self._players_box)

        # -- Buttons --
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Start Game")
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        outer.addWidget(self._button_box)

        # Build initial rows then connect spin so it rebuilds on change.
        self._rebuild_player_rows(self._player_spin.value())
        self._player_spin.valueChanged.connect(self._rebuild_player_rows)

    # ------------------------------------------------------------------

    def _rebuild_player_rows(self, num_players: int) -> None:
        """Rebuild the per-player rows to match the chosen player count."""
        # Remove old widgets from layout.
        while self._players_layout.count():
            item = self._players_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._player_rows.clear()

        # Header labels.
        self._players_layout.addWidget(QLabel("Player"), 0, 0)
        self._players_layout.addWidget(QLabel("Type"), 0, 1)
        self._players_layout.addWidget(QLabel("Model"), 0, 2)

        for i in range(num_players):
            row: dict = {}
            grid_row = i + 1

            # Player label.
            lbl = QLabel(f"Player {i + 1}")
            lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            self._players_layout.addWidget(lbl, grid_row, 0)

            # Human / AI toggle.
            type_widget = QWidget()
            type_layout = QHBoxLayout(type_widget)
            type_layout.setContentsMargins(0, 0, 0, 0)
            type_layout.setSpacing(6)

            human_rb = QRadioButton("Human")
            ai_rb = QRadioButton("AI")
            # Player 0 defaults to Human; all others default to AI.
            if i == 0:
                human_rb.setChecked(True)
            else:
                ai_rb.setChecked(True)

            btn_group = QButtonGroup(type_widget)
            btn_group.addButton(human_rb, 0)
            btn_group.addButton(ai_rb, 1)
            type_layout.addWidget(human_rb)
            type_layout.addWidget(ai_rb)
            self._players_layout.addWidget(type_widget, grid_row, 1)

            # Model row (load button + filename label).
            model_widget = QWidget()
            model_layout = QHBoxLayout(model_widget)
            model_layout.setContentsMargins(0, 0, 0, 0)
            model_layout.setSpacing(6)

            load_btn = QPushButton("Load...")
            load_btn.setFixedWidth(60)
            path_lbl = QLabel("")
            path_lbl.setFont(QFont("Consolas", 8))
            path_lbl.setWordWrap(False)
            model_layout.addWidget(load_btn)
            model_layout.addWidget(path_lbl, stretch=1)
            self._players_layout.addWidget(model_widget, grid_row, 2)

            row = {
                "human_rb": human_rb,
                "ai_rb": ai_rb,
                "btn_group": btn_group,
                "load_btn": load_btn,
                "path_lbl": path_lbl,
                "checkpoint_path": None,
                "model_widget": model_widget,
            }
            self._player_rows.append(row)

            # Show/hide model picker based on current selection.
            self._update_row_visibility(row)

            # Connect signals (use default-arg capture for closure).
            btn_group.buttonToggled.connect(
                lambda _btn, _checked, r=row: self._on_type_toggled(r)
            )
            load_btn.clicked.connect(lambda _=False, r=row: self._on_load_model(r))

        self._validate_ok_button()

    def _update_row_visibility(self, row: dict) -> None:
        row["model_widget"].setVisible(row["ai_rb"].isChecked())

    def _on_type_toggled(self, row: dict) -> None:
        self._update_row_visibility(row)
        self._validate_ok_button()

    def _on_load_model(self, row: dict) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AI Model Checkpoint",
            "",
            "Model files (*.zip);;All files (*.*)",
        )
        if path:
            row["checkpoint_path"] = path
            row["path_lbl"].setText(os.path.basename(path))
            self._validate_ok_button()

    def _validate_ok_button(self) -> None:
        ok = True
        for row in self._player_rows:
            if row["ai_rb"].isChecked() and not row["checkpoint_path"]:
                ok = False
                break
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    # ------------------------------------------------------------------

    def get_config(self) -> GameSetupConfig:
        """Return the configured GameSetupConfig."""
        configs = []
        for i, row in enumerate(self._player_rows):
            configs.append(PlayerConfig(
                player_id=i,
                is_human=row["human_rb"].isChecked(),
                checkpoint_path=row["checkpoint_path"] if row["ai_rb"].isChecked() else None,
            ))
        return GameSetupConfig(player_configs=configs)

    def get_num_players(self) -> int:
        """Convenience accessor kept for backwards compatibility."""
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

        # Spectate controls (hidden unless all-AI game).
        spectate_row = QHBoxLayout()
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(90)
        self._pause_btn.setVisible(False)
        self._pause_callback: Optional[Callable[[], None]] = None
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        spectate_row.addStretch()
        spectate_row.addWidget(self._pause_btn)
        main_layout.addLayout(spectate_row)

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
            config = dialog.get_config()
            self._message_log.clear()
            self.add_message(f"Starting new game with {config.num_players} players...")
            if hasattr(self, '_new_game_callback') and self._new_game_callback:
                self._new_game_callback(config)

    def set_new_game_callback(self, callback: Callable[["GameSetupConfig"], None]) -> None:
        """Set the callback for starting a new game."""
        self._new_game_callback = callback

    def set_ai_players(self, ai_player_ids: set[int]) -> None:
        """Mark which player IDs are AI-controlled (for badge display)."""
        self._player_info.set_ai_players(ai_player_ids)

    def set_spectate_mode(self, enabled: bool) -> None:
        """Show or hide the Pause/Resume button for spectate (all-AI) mode."""
        self._pause_btn.setVisible(enabled)

    def set_pause_callback(self, callback: Callable[[], None]) -> None:
        """Set the callback invoked when Pause/Resume is clicked."""
        self._pause_callback = callback

    def update_pause_label(self, paused: bool) -> None:
        """Update Pause/Resume button text to reflect current state."""
        self._pause_btn.setText("Resume" if paused else "Pause")

    def _on_pause_clicked(self) -> None:
        if hasattr(self, '_pause_callback') and self._pause_callback:
            self._pause_callback()

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
