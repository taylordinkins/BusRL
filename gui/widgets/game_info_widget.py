"""Game info widget for displaying global game state."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush

from core.game_state import GameState
from core.constants import (
    BuildingType, Phase, TOTAL_PASSENGERS, TOTAL_BUILDINGS_PER_TYPE
)

from gui.constants import TIME_CLOCK_COLORS, BUILDING_COLORS


class TimeClockWidget(QFrame):
    """Widget showing the time clock position."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_position: BuildingType = BuildingType.HOUSE
        self._stones_remaining: int = 5

        self.setFixedSize(220, 95)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

    def set_position(self, position: BuildingType, stones_remaining: int) -> None:
        """Update the clock position and stones."""
        self._current_position = position
        self._stones_remaining = stones_remaining
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the time clock."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw title
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#333333"))
        painter.drawText(10, 18, "Time Clock")

        # Draw clock positions
        positions = [BuildingType.HOUSE, BuildingType.OFFICE, BuildingType.PUB]
        names = ["House", "Office", "Pub"]

        x_start = 20
        y = 48
        spacing = 65

        for i, (building_type, name) in enumerate(zip(positions, names)):
            x = x_start + i * spacing
            color = BUILDING_COLORS[building_type]

            is_current = building_type == self._current_position

            # Draw indicator circle
            if is_current:
                painter.setPen(QPen(QColor("#000000"), 3))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(x, y - 15, 30, 30)
            else:
                painter.setPen(QPen(color.darker(120), 2))
                painter.setBrush(QBrush(color.lighter(150)))
                painter.drawEllipse(x + 5, y - 10, 20, 20)

            # Draw label
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QColor("#333333"))
            label_x = x + (15 if is_current else 15) - len(name) * 2
            painter.drawText(label_x, y + 25, name)

        # Draw time stones with more space
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QColor("#333333"))
        painter.drawText(10, 90, f"Time Stones: {self._stones_remaining}")


class PhaseIndicatorWidget(QFrame):
    """Widget showing the current game phase."""

    PHASE_NAMES = {
        Phase.SETUP_BUILDINGS: "Setup: Buildings",
        Phase.SETUP_RAILS_FORWARD: "Setup: Rails (Fwd)",
        Phase.SETUP_RAILS_REVERSE: "Setup: Rails (Rev)",
        Phase.CHOOSING_ACTIONS: "Choosing Actions",
        Phase.RESOLVING_ACTIONS: "Resolving Actions",
        Phase.CLEANUP: "Cleanup",
        Phase.GAME_OVER: "Game Over",
    }

    PHASE_COLORS = {
        Phase.SETUP_BUILDINGS: QColor("#FFE0B2"),
        Phase.SETUP_RAILS_FORWARD: QColor("#FFE0B2"),
        Phase.SETUP_RAILS_REVERSE: QColor("#FFE0B2"),
        Phase.CHOOSING_ACTIONS: QColor("#C8E6C9"),
        Phase.RESOLVING_ACTIONS: QColor("#BBDEFB"),
        Phase.CLEANUP: QColor("#E1BEE7"),
        Phase.GAME_OVER: QColor("#FFCDD2"),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._phase = Phase.SETUP_BUILDINGS
        self._round = 1

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Round label
        self._round_label = QLabel("Round 1")
        self._round_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self._round_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._round_label)

        # Phase label
        self._phase_label = QLabel("Setup: Buildings")
        self._phase_label.setFont(QFont("Arial", 11))
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._phase_label)

        self._update_style()

    def set_phase(self, phase: Phase, round_number: int) -> None:
        """Update the phase and round display."""
        self._phase = phase
        self._round = round_number

        self._round_label.setText(f"Round {round_number}")
        self._phase_label.setText(self.PHASE_NAMES.get(phase, phase.value))
        self._update_style()

    def _update_style(self) -> None:
        """Update the background color based on phase."""
        color = self.PHASE_COLORS.get(self._phase, QColor("#FFFFFF"))
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {color.name()};
                border: 2px solid {color.darker(120).name()};
                border-radius: 4px;
            }}
        """)


class BoardSummaryWidget(QFrame):
    """Widget showing summary board statistics and remaining component supplies."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(x := 10) # Using a slightly larger spacing for clarity

        # Column 1: Supply counts
        layout.addWidget(QLabel("Supply Rem:"), 0, 0, 1, 2)
        
        # Passengers
        psgr_label = QLabel("Passengers:")
        layout.addWidget(psgr_label, 1, 0)
        self._passengers_label = QLabel("15")
        self._passengers_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self._passengers_label, 1, 1)

        # Buildings by type
        layout.addWidget(QLabel("Houses:"), 2, 0)
        self._houses_label = QLabel("30")
        layout.addWidget(self._houses_label, 2, 1)

        layout.addWidget(QLabel("Offices:"), 3, 0)
        self._offices_label = QLabel("30")
        layout.addWidget(self._offices_label, 3, 1)

        layout.addWidget(QLabel("Pubs:"), 4, 0)
        self._pubs_label = QLabel("30")
        layout.addWidget(self._pubs_label, 4, 1)

        # Column 2: Board status
        layout.addWidget(QLabel("Board:"), 0, 2, 1, 2)

        layout.addWidget(QLabel("Empty Slots:"), 1, 2)
        self._empty_slots_label = QLabel("0")
        layout.addWidget(self._empty_slots_label, 1, 3)

        layout.addWidget(QLabel("Rails:"), 2, 2)
        self._rails_label = QLabel("0")
        layout.addWidget(self._rails_label, 2, 3)

        layout.addWidget(QLabel("Total Bldgs:"), 3, 2)
        self._total_buildings_label = QLabel("0")
        layout.addWidget(self._total_buildings_label, 3, 3)

    def update_from_state(self, state: GameState) -> None:
        """Update statistics from game state."""
        # Remaining Passengers
        on_board = state.passenger_manager.count()
        remaining_psgrs = max(0, TOTAL_PASSENGERS - on_board)
        self._passengers_label.setText(str(remaining_psgrs))

        # Remaining Buildings by type
        houses_rem = max(0, TOTAL_BUILDINGS_PER_TYPE - state.board.get_building_count(BuildingType.HOUSE))
        offices_rem = max(0, TOTAL_BUILDINGS_PER_TYPE - state.board.get_building_count(BuildingType.OFFICE))
        pubs_rem = max(0, TOTAL_BUILDINGS_PER_TYPE - state.board.get_building_count(BuildingType.PUB))
        
        self._houses_label.setText(str(houses_rem))
        self._offices_label.setText(str(offices_rem))
        self._pubs_label.setText(str(pubs_rem))

        # Count total buildings and empty slots
        total_buildings = 0
        empty_slots = 0
        for node in state.board.nodes.values():
            for slot in node.building_slots:
                if slot.building:
                    total_buildings += 1
                else:
                    empty_slots += 1
        
        self._total_buildings_label.setText(str(total_buildings))
        self._empty_slots_label.setText(str(empty_slots))

        # Count rails
        rails = sum(len(e.rail_segments) for e in state.board.edges.values())
        self._rails_label.setText(str(rails))


class GameInfoWidget(QWidget):
    """Widget showing global game information."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(16)

        # Phase indicator
        self._phase_widget = PhaseIndicatorWidget()
        layout.addWidget(self._phase_widget)

        # Time clock
        self._clock_widget = TimeClockWidget()
        layout.addWidget(self._clock_widget)

        # Board summary
        self._summary_widget = BoardSummaryWidget()
        layout.addWidget(self._summary_widget)

        layout.addStretch()

    def set_state(self, state: GameState) -> None:
        """Update all displays from game state."""
        self._phase_widget.set_phase(
            state.phase,
            state.global_state.round_number
        )
        self._clock_widget.set_position(
            state.global_state.time_clock_position,
            state.global_state.time_stones_remaining
        )
        self._summary_widget.update_from_state(state)
