"""Dialog windows for the Bus GUI."""

from __future__ import annotations

from typing import Optional, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGridLayout, QMessageBox, QWidget, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor

from core.game_state import GameState
from core.player import Player
from engine.driver import ActionChoice

from gui.constants import PLAYER_COLORS, BUILDING_COLORS


class ChoiceDialog(QDialog):
    """Dialog for presenting a list of choices to the player."""

    def __init__(
        self,
        title: str,
        message: str,
        choices: list[ActionChoice],
        allow_cancel: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self._choices = choices
        self._selected_choice: Optional[ActionChoice] = None
        self._allow_cancel = allow_cancel

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Message
        msg_label = QLabel(message)
        msg_label.setFont(QFont("Arial", 11))
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        # Choices list
        self._list = QListWidget()
        self._list.setFont(QFont("Arial", 10))
        self._list.setAlternatingRowColors(True)

        for choice in choices:
            item = QListWidgetItem(f"{choice.index}. {choice.description}")
            item.setData(Qt.ItemDataRole.UserRole, choice)
            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        self._ok_button = QPushButton("Select")
        self._ok_button.setEnabled(False)
        self._ok_button.clicked.connect(self._on_ok)
        button_layout.addWidget(self._ok_button)

        if allow_cancel:
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        # Select first item by default
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_selection_changed(self, current, previous) -> None:
        """Handle selection change."""
        self._ok_button.setEnabled(current is not None)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double click on item."""
        self._selected_choice = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_ok(self) -> None:
        """Handle OK button click."""
        current = self._list.currentItem()
        if current:
            self._selected_choice = current.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def get_selected_choice(self) -> Optional[ActionChoice]:
        """Get the selected choice."""
        return self._selected_choice


class ConfirmDialog(QDialog):
    """Simple yes/no confirmation dialog."""

    def __init__(
        self,
        title: str,
        message: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Message
        msg_label = QLabel(message)
        msg_label.setFont(QFont("Arial", 11))
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class GameOverDialog(QDialog):
    """Dialog showing game over results and final scores."""

    def __init__(
        self,
        state: GameState,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Game Over")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("GAME OVER")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Sort players by final score
        sorted_players = sorted(
            state.players,
            key=lambda p: p.score - p.time_stones,
            reverse=True
        )

        # Winner announcement
        winner = sorted_players[0]
        winner_color = PLAYER_COLORS[winner.player_id % len(PLAYER_COLORS)]
        winner_label = QLabel(f"Player {winner.player_id} Wins!")
        winner_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        winner_label.setStyleSheet(f"color: {winner_color.name()};")
        winner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(winner_label)

        # Scores frame
        scores_frame = QFrame()
        scores_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        scores_layout = QGridLayout(scores_frame)
        scores_layout.setSpacing(8)

        # Header
        scores_layout.addWidget(QLabel("Rank"), 0, 0)
        scores_layout.addWidget(QLabel("Player"), 0, 1)
        scores_layout.addWidget(QLabel("Score"), 0, 2)
        scores_layout.addWidget(QLabel("Time Stones"), 0, 3)
        scores_layout.addWidget(QLabel("Final Score"), 0, 4)

        for col in range(5):
            header = scores_layout.itemAtPosition(0, col).widget()
            header.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        # Player rows
        for rank, player in enumerate(sorted_players, 1):
            color = PLAYER_COLORS[player.player_id % len(PLAYER_COLORS)]
            final_score = player.score - player.time_stones

            rank_label = QLabel(str(rank))
            scores_layout.addWidget(rank_label, rank, 0)

            player_label = QLabel(f"Player {player.player_id}")
            player_label.setStyleSheet(f"color: {color.name()};")
            player_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            scores_layout.addWidget(player_label, rank, 1)

            scores_layout.addWidget(QLabel(str(player.score)), rank, 2)

            stones_label = QLabel(f"-{player.time_stones}" if player.time_stones > 0 else "0")
            if player.time_stones > 0:
                stones_label.setStyleSheet("color: red;")
            scores_layout.addWidget(stones_label, rank, 3)

            final_label = QLabel(str(final_score))
            final_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            scores_layout.addWidget(final_label, rank, 4)

        layout.addWidget(scores_frame)

        # Game statistics
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        stats_layout = QGridLayout(stats_frame)

        stats_layout.addWidget(QLabel("Rounds Played:"), 0, 0)
        stats_layout.addWidget(QLabel(str(state.global_state.round_number)), 0, 1)

        stats_layout.addWidget(QLabel("Total Passengers:"), 1, 0)
        stats_layout.addWidget(QLabel(str(state.passenger_manager.count())), 1, 1)

        buildings = sum(
            1 for node in state.board.nodes.values()
            for slot in node.building_slots if slot.building
        )
        stats_layout.addWidget(QLabel("Buildings Placed:"), 2, 0)
        stats_layout.addWidget(QLabel(str(buildings)), 2, 1)

        layout.addWidget(stats_frame)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


class PassengerDistributionDialog(QDialog):
    """Dialog for choosing how to distribute passengers to train stations."""

    def __init__(
        self,
        total_passengers: int,
        station_ids: list[int],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Distribute Passengers")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._total = total_passengers
        self._station_ids = station_ids
        self._distribution: dict[int, int] = {sid: 0 for sid in station_ids}

        layout = QVBoxLayout(self)

        # Instructions
        instr = QLabel(f"Distribute {total_passengers} passenger(s) to train stations:")
        instr.setFont(QFont("Arial", 11))
        layout.addWidget(instr)

        # Station controls
        self._station_labels: dict[int, QLabel] = {}
        self._remaining_label = QLabel(f"Remaining: {total_passengers}")
        self._remaining_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        for station_id in station_ids:
            row_layout = QHBoxLayout()

            row_layout.addWidget(QLabel(f"Station {station_id}:"))

            minus_btn = QPushButton("-")
            minus_btn.setFixedWidth(30)
            minus_btn.clicked.connect(lambda checked, s=station_id: self._adjust(s, -1))
            row_layout.addWidget(minus_btn)

            count_label = QLabel("0")
            count_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            count_label.setMinimumWidth(30)
            count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._station_labels[station_id] = count_label
            row_layout.addWidget(count_label)

            plus_btn = QPushButton("+")
            plus_btn.setFixedWidth(30)
            plus_btn.clicked.connect(lambda checked, s=station_id: self._adjust(s, 1))
            row_layout.addWidget(plus_btn)

            row_layout.addStretch()
            layout.addLayout(row_layout)

        layout.addWidget(self._remaining_label)

        # Buttons
        self._ok_button = QPushButton("Confirm")
        self._ok_button.setEnabled(False)
        self._ok_button.clicked.connect(self.accept)
        layout.addWidget(self._ok_button)

    def _adjust(self, station_id: int, delta: int) -> None:
        """Adjust the count for a station."""
        current = self._distribution[station_id]
        new_value = current + delta

        # Check bounds
        if new_value < 0:
            return

        remaining = self._total - sum(self._distribution.values())
        if delta > 0 and remaining <= 0:
            return

        self._distribution[station_id] = new_value
        self._station_labels[station_id].setText(str(new_value))

        # Update remaining
        remaining = self._total - sum(self._distribution.values())
        self._remaining_label.setText(f"Remaining: {remaining}")

        # Enable OK button when all distributed
        self._ok_button.setEnabled(remaining == 0)

    def get_distribution(self) -> dict[int, int]:
        """Get the final distribution."""
        return self._distribution.copy()


class BuildingPlacementDialog(QDialog):
    """Dialog for choosing building placement."""

    def __init__(
        self,
        valid_placements: list[dict],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Place Building")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(350)

        self._placements = valid_placements
        self._selected: Optional[dict] = None

        layout = QVBoxLayout(self)

        # Instructions
        instr = QLabel("Choose where and what to build:")
        instr.setFont(QFont("Arial", 11))
        layout.addWidget(instr)

        # Choices list
        self._list = QListWidget()
        self._list.setFont(QFont("Arial", 10))

        for placement in valid_placements:
            building = placement["building_type"]
            node_id = placement["node_id"]
            slot_idx = placement["slot_index"]
            zone = placement.get("zone", "?")

            # Handle building_type as either BuildingType enum or string
            if hasattr(building, 'value'):
                building_name = building.value.upper()
                building_key = building
            else:
                building_name = str(building).upper()
                # Try to get the BuildingType enum for color lookup
                try:
                    from core.constants import BuildingType
                    building_key = BuildingType(building)
                except (ValueError, KeyError):
                    building_key = None

            # Get building color for display
            text = f"Node {node_id}, Slot {slot_idx} (Zone {zone}): {building_name}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, placement)

            # Color code by building type
            color = BUILDING_COLORS.get(building_key, QColor("#FFFFFF"))
            item.setBackground(color.lighter(150))

            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        self._ok_button = QPushButton("Place")
        self._ok_button.setEnabled(False)
        self._ok_button.clicked.connect(self._on_ok)
        button_layout.addWidget(self._ok_button)

        layout.addLayout(button_layout)

        # Select first item
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_selection_changed(self, current, previous) -> None:
        self._ok_button.setEnabled(current is not None)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_ok(self) -> None:
        current = self._list.currentItem()
        if current:
            self._selected = current.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def get_selected(self) -> Optional[dict]:
        return self._selected


class RailPlacementDialog(QDialog):
    """Dialog for choosing rail segment placement."""

    def __init__(
        self,
        valid_placements: list[dict],
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Extend Rail Network")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self._placements = valid_placements
        self._selected: Optional[dict] = None

        layout = QVBoxLayout(self)

        # Instructions
        instr = QLabel("Choose where to extend your rail network:")
        instr.setFont(QFont("Arial", 11))
        layout.addWidget(instr)

        # Choices list
        self._list = QListWidget()
        self._list.setFont(QFont("Arial", 10))

        for placement in valid_placements:
            edge = placement["edge_id"]
            from_node = placement["from_endpoint"]
            text = f"From node {from_node}: Edge {edge[0]} -- {edge[1]}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, placement)
            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        self._ok_button = QPushButton("Place Rail")
        self._ok_button.setEnabled(False)
        self._ok_button.clicked.connect(self._on_ok)
        button_layout.addWidget(self._ok_button)

        layout.addLayout(button_layout)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_selection_changed(self, current, previous) -> None:
        self._ok_button.setEnabled(current is not None)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_ok(self) -> None:
        current = self._list.currentItem()
        if current:
            self._selected = current.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def get_selected(self) -> Optional[dict]:
        return self._selected


class VrrooommDialog(QDialog):
    """Dialog for choosing passenger deliveries during Vrroomm!"""

    def __init__(
        self,
        valid_deliveries: list[dict],
        deliveries_remaining: int,
        buses: int,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Vrroomm! - Deliver Passengers")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._deliveries = valid_deliveries
        self._selected: Optional[dict] = None
        self._skip = False

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"Deliveries remaining: {deliveries_remaining} (Buses: {buses})")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(header)

        # Instructions
        instr = QLabel("Choose a passenger to deliver, or skip:")
        layout.addWidget(instr)

        # Choices list
        self._list = QListWidget()
        self._list.setFont(QFont("Arial", 10))

        for delivery in valid_deliveries:
            p_id = delivery["passenger_id"]
            from_node = delivery["from_node"]
            to_node = delivery["to_node"]
            slot_idx = delivery["building_slot_index"]
            text = f"Passenger {p_id}: Node {from_node} → Node {to_node} (slot {slot_idx})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, delivery)
            self._list.addItem(item)

        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        # Buttons
        button_layout = QHBoxLayout()

        self._deliver_button = QPushButton("Deliver")
        self._deliver_button.setEnabled(False)
        self._deliver_button.clicked.connect(self._on_deliver)
        button_layout.addWidget(self._deliver_button)

        skip_button = QPushButton("Skip Remaining")
        skip_button.clicked.connect(self._on_skip)
        button_layout.addWidget(skip_button)

        layout.addLayout(button_layout)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _on_selection_changed(self, current, previous) -> None:
        self._deliver_button.setEnabled(current is not None)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_deliver(self) -> None:
        current = self._list.currentItem()
        if current:
            self._selected = current.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def _on_skip(self) -> None:
        self._skip = True
        self.accept()

    def get_selected(self) -> Optional[dict]:
        return self._selected

    def is_skip(self) -> bool:
        return self._skip


class TimeClockDialog(QDialog):
    """Dialog for time clock action choice."""

    def __init__(
        self,
        current_position: str,
        stones_remaining: int,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Time Clock")
        self.setModal(True)
        self.setMinimumWidth(350)

        self._choice: Optional[str] = None

        layout = QVBoxLayout(self)

        # Current state
        state_label = QLabel(
            f"Current position: {current_position.upper()}\n"
            f"Time stones remaining: {stones_remaining}"
        )
        state_label.setFont(QFont("Arial", 11))
        layout.addWidget(state_label)

        # Choices
        advance_button = QPushButton("Advance Clock")
        advance_button.setFont(QFont("Arial", 12))
        advance_button.clicked.connect(lambda: self._select("advance_clock"))
        layout.addWidget(advance_button)

        if stones_remaining > 0:
            stop_button = QPushButton(f"Stop Clock (take time stone, -1 point)")
            stop_button.setFont(QFont("Arial", 12))
            stop_button.clicked.connect(lambda: self._select("stop_clock"))
            layout.addWidget(stop_button)

    def _select(self, choice: str) -> None:
        self._choice = choice
        self.accept()

    def get_choice(self) -> Optional[str]:
        return self._choice

class CleanupDialog(QDialog):
    """Dialog to confirm cleanup and moving to the next round."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("Round End")
        self.setModal(True)
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        msg = QLabel("Round Complete!")
        msg.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg)

        info = QLabel("All actions have been resolved. Ready to perform cleanup and start the next round?")
        info.setFont(QFont("Arial", 11))
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        # Buttons
        button_layout = QHBoxLayout()
        confirm_btn = QPushButton("Perform Cleanup & Start Next Round")
        confirm_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        confirm_btn.setMinimumHeight(40)
        confirm_btn.clicked.connect(self.accept)
        button_layout.addWidget(confirm_btn)

        layout.addLayout(button_layout)
