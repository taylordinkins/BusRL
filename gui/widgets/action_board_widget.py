"""Action board widget for displaying marker placements."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QSizePolicy, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette, QColor

from core.game_state import GameState
from core.action_board import ActionBoard, ActionArea, ActionSlot, get_slot_labels_for_resolution
from core.constants import ActionAreaType, ACTION_RESOLUTION_ORDER, MIN_MARKERS_PER_ROUND

from gui.constants import PLAYER_COLORS, PLAYER_COLOR_NAMES, ACTION_AREA_NAMES


class SlotWidget(QFrame):
    """Widget representing a single action slot."""

    clicked = Signal(str)  # slot label

    def __init__(
        self,
        label: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._label = label
        self._player_id: Optional[int] = None
        self._is_highlighted = False

        self.setFixedSize(40, 40)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Slot label at top
        self._label_widget = QLabel(label)
        self._label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_widget.setFont(QFont("Arial", 8))
        layout.addWidget(self._label_widget)

        # Player marker indicator
        self._marker_widget = QLabel("")
        self._marker_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._marker_widget.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self._marker_widget)

        self._update_appearance()

    def set_player(self, player_id: Optional[int]) -> None:
        """Set the player who has a marker in this slot."""
        self._player_id = player_id
        self._update_appearance()

    def set_highlighted(self, highlighted: bool) -> None:
        """Set whether this slot is highlighted (valid placement)."""
        self._is_highlighted = highlighted
        self._update_appearance()

    def _update_appearance(self) -> None:
        """Update the visual appearance based on state."""
        if self._player_id is not None:
            # Slot is occupied
            color = PLAYER_COLORS[self._player_id % len(PLAYER_COLORS)]
            self._marker_widget.setText(f"P{self._player_id}")
            self._marker_widget.setStyleSheet(f"color: {color.name()};")
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {color.lighter(180).name()};
                    border: 2px solid {color.name()};
                }}
            """)
        elif self._is_highlighted:
            # Slot is available for placement
            self._marker_widget.setText("")
            self.setStyleSheet("""
                QFrame {
                    background-color: #FFFFD0;
                    border: 2px solid #FFD700;
                }
            """)
        else:
            # Empty slot
            self._marker_widget.setText("")
            self.setStyleSheet("""
                QFrame {
                    background-color: #F0F0F0;
                    border: 2px solid #CCCCCC;
                }
            """)

    def mousePressEvent(self, event) -> None:
        """Handle click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._label)


class ActionAreaWidget(QFrame):
    """Widget representing a single action area with its slots."""

    slot_clicked = Signal(str, str)  # area_type value, slot label

    def __init__(
        self,
        area_type: ActionAreaType,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self._area_type = area_type
        self._slots: dict[str, SlotWidget] = {}

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Area name label
        name = ACTION_AREA_NAMES.get(area_type.value, area_type.value)
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        name_label.setMinimumWidth(120)
        layout.addWidget(name_label)

        # Spacer
        layout.addStretch()

        # Slots container
        slots_layout = QHBoxLayout()
        slots_layout.setSpacing(4)

        # Get slot count and create slots in resolution order
        from core.action_board import _get_max_slots
        num_slots = _get_max_slots(area_type)
        labels = get_slot_labels_for_resolution(area_type, num_slots)

        for label in labels:
            slot_widget = SlotWidget(label)
            slot_widget.clicked.connect(
                lambda lbl, at=area_type: self.slot_clicked.emit(at.value, lbl)
            )
            self._slots[label] = slot_widget
            slots_layout.addWidget(slot_widget)

        layout.addLayout(slots_layout)

    def update_from_area(self, area: ActionArea) -> None:
        """Update slot displays from an ActionArea."""
        for label, slot_widget in self._slots.items():
            slot = area.slots.get(label)
            if slot:
                slot_widget.set_player(slot.player_id)
            else:
                slot_widget.set_player(None)

    def set_highlighted_slot(self, slot_label: Optional[str]) -> None:
        """Highlight a specific slot (for valid placements)."""
        for label, slot_widget in self._slots.items():
            slot_widget.set_highlighted(label == slot_label)


class ActionBoardWidget(QWidget):
    """Widget for displaying the complete action board."""

    slot_clicked = Signal(str, str)  # area_type value, slot label
    pass_clicked = Signal()  # Emitted when Pass button is clicked

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._state: Optional[GameState] = None
        self._area_widgets: dict[ActionAreaType, ActionAreaWidget] = {}
        self._highlighted_areas: dict[ActionAreaType, str] = {}  # area -> next slot label
        self._pass_enabled = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("Action Board")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Create area widgets in resolution order
        for area_type in ACTION_RESOLUTION_ORDER:
            area_widget = ActionAreaWidget(area_type)
            area_widget.slot_clicked.connect(self._on_slot_clicked)
            self._area_widgets[area_type] = area_widget
            layout.addWidget(area_widget)

        layout.addStretch()

        # Bus count display
        self._bus_display = BusCountWidget()
        layout.addWidget(self._bus_display)

        # Pass button (hidden by default)
        self._pass_button = QPushButton("PASS (End Marker Placement)")
        self._pass_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self._pass_button.setMinimumHeight(40)
        self._pass_button.setStyleSheet("""
            QPushButton {
                background-color: #FFF3E0;
                border: 2px solid #FF9800;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #FFE0B2;
            }
            QPushButton:pressed {
                background-color: #FFCC80;
            }
            QPushButton:disabled {
                background-color: #E0E0E0;
                border: 2px solid #BDBDBD;
                color: #9E9E9E;
            }
        """)
        self._pass_button.clicked.connect(self._on_pass_clicked)
        self._pass_button.setVisible(False)
        layout.addWidget(self._pass_button)

    def set_state(self, state: GameState) -> None:
        """Update the display from game state."""
        self._state = state

        for area_type, area_widget in self._area_widgets.items():
            area = state.action_board.get_area(area_type)
            area_widget.update_from_area(area)

        self._bus_display.update_from_state(state)

    def set_available_areas(self, areas: list[ActionAreaType]) -> None:
        """Highlight the next available slot in each available area."""
        self._highlighted_areas.clear()

        if self._state is None:
            return

        for area_type in areas:
            area = self._state.action_board.get_area(area_type)
            next_slot = area.get_next_available_slot()
            if next_slot:
                self._highlighted_areas[area_type] = next_slot.label

        # Update highlights
        for area_type, area_widget in self._area_widgets.items():
            if area_type in self._highlighted_areas:
                area_widget.set_highlighted_slot(self._highlighted_areas[area_type])
            else:
                area_widget.set_highlighted_slot(None)

    def set_pass_enabled(self, enabled: bool, can_pass: bool = False, label: Optional[str] = None) -> None:
        """Show/hide and enable/disable the Pass button.

        Args:
            enabled: Whether to show the Pass button
            can_pass: Whether the player can actually pass (met minimum markers)
            label: Optional custom label for the button
        """
        self._pass_enabled = enabled and can_pass
        self._pass_button.setVisible(enabled)
        self._pass_button.setEnabled(can_pass)

        if label:
            self._pass_button.setText(label)
        elif enabled and not can_pass:
            self._pass_button.setText(f"PASS (need {MIN_MARKERS_PER_ROUND} markers first)")
        else:
            self._pass_button.setText("PASS (End Marker Placement)")

    def clear_highlights(self) -> None:
        """Clear all slot highlights."""
        self._highlighted_areas.clear()
        for area_widget in self._area_widgets.values():
            area_widget.set_highlighted_slot(None)
        self._pass_button.setVisible(False)

    def _on_slot_clicked(self, area_type_value: str, slot_label: str) -> None:
        """Handle slot click."""
        self.slot_clicked.emit(area_type_value, slot_label)

    def _on_pass_clicked(self) -> None:
        """Handle Pass button click."""
        self.pass_clicked.emit()


class BusCountWidget(QFrame):
    """Widget showing bus counts for each player."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(16)

        # Title
        title = QLabel("Buses:")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self._layout.addWidget(title)

        # Player bus counts
        self._bus_labels: list[QLabel] = []

    def update_from_state(self, state: GameState) -> None:
        """Update bus counts from game state."""
        # Clear existing labels
        for label in self._bus_labels:
            self._layout.removeWidget(label)
            label.deleteLater()
        self._bus_labels.clear()

        # Create new labels
        max_buses = max(p.buses for p in state.players)

        for player in state.players:
            color = PLAYER_COLORS[player.player_id % len(PLAYER_COLORS)]

            label = QLabel(f"P{player.player_id}: {player.buses}")
            label.setFont(QFont("Arial", 10))
            label.setStyleSheet(f"color: {color.name()};")

            # Highlight max buses
            if player.buses == max_buses:
                label.setFont(QFont("Arial", 10, QFont.Weight.Bold))

            self._bus_labels.append(label)
            self._layout.addWidget(label)

        # Add M#oB indicator
        mob_label = QLabel(f"(M#oB: {max_buses})")
        mob_label.setFont(QFont("Arial", 9))
        mob_label.setStyleSheet("color: #666666;")
        self._bus_labels.append(mob_label)
        self._layout.addWidget(mob_label)

        self._layout.addStretch()
