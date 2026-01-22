"""Player info widget for displaying player status and resources."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

from core.game_state import GameState
from core.player import Player

from gui.constants import PLAYER_COLORS, PLAYER_COLOR_NAMES


class PlayerCardWidget(QFrame):
    """Widget showing a single player's status."""

    def __init__(self, player_id: int, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._player_id = player_id
        self._is_current = False
        self._is_starting = False

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setMinimumWidth(100) # Slightly narrower

        color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
        self._base_color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4) # Tighter margins
        layout.setSpacing(0) # Minimal spacing

        # Player header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)

        self._name_label = QLabel(f"P{player_id}") # Shorter name
        self._name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self._name_label.setStyleSheet(f"color: {color.name()};")
        header_layout.addWidget(self._name_label)

        header_layout.addStretch()

        # Current player indicator
        self._current_indicator = QLabel("")
        self._current_indicator.setFont(QFont("Arial", 9))
        header_layout.addWidget(self._current_indicator)

        layout.addLayout(header_layout)

        # Stats grid
        stats_layout = QGridLayout()
        stats_layout.setSpacing(2)

        # Score
        self._score_label = QLabel("0")
        self._score_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        stats_layout.addWidget(self._score_label, 0, 0)
        stats_layout.addWidget(QLabel("pts"), 0, 1)

        # Resource icons/short labels
        stats_layout.addWidget(QLabel("B:"), 1, 0)
        self._buses_label = QLabel("1")
        stats_layout.addWidget(self._buses_label, 1, 1)

        stats_layout.addWidget(QLabel("M:"), 2, 0)
        self._markers_label = QLabel("20")
        stats_layout.addWidget(self._markers_label, 2, 1)

        stats_layout.addWidget(QLabel("R:"), 3, 0)
        self._rails_label = QLabel("25")
        stats_layout.addWidget(self._rails_label, 3, 1)

        stats_layout.addWidget(QLabel("T:"), 4, 0)
        self._stones_label = QLabel("0")
        stats_layout.addWidget(self._stones_label, 4, 1)

        layout.addLayout(stats_layout)

        # Status label (passed, etc.)
        self._status_label = QLabel("")
        self._status_label.setFont(QFont("Arial", 9))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        self._update_style()

    def update_from_player(
        self,
        player: Player,
        is_current: bool,
        is_starting: bool
    ) -> None:
        """Update display from player state."""
        self._is_current = is_current
        self._is_starting = is_starting

        self._score_label.setText(str(player.score))
        self._buses_label.setText(str(player.buses))
        self._markers_label.setText(str(player.action_markers_remaining))
        self._rails_label.setText(str(player.rail_segments_remaining))
        self._stones_label.setText(str(player.time_stones))

        # Update indicators
        indicators = []
        if is_current:
            indicators.append("TURN")
        if is_starting:
            indicators.append("START")
        self._current_indicator.setText(" ".join(indicators))

        # Status
        if player.has_passed:
            self._status_label.setText("[PASSED]")
            self._status_label.setStyleSheet("color: #888888;")
        else:
            self._status_label.setText("")

        self._update_style()

    def _update_style(self) -> None:
        """Update the frame style based on state."""
        color = self._base_color

        if self._is_current:
            # Highlight current player
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {color.lighter(180).name()};
                    border: 3px solid {color.name()};
                    border-radius: 4px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #FAFAFA;
                    border: 2px solid {color.lighter(150).name()};
                    border-radius: 4px;
                }}
            """)


class PlayerInfoWidget(QWidget):
    """Widget showing all players' information."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._state: Optional[GameState] = None
        self._player_cards: list[PlayerCardWidget] = []

        self._layout = QHBoxLayout(self) # Back to horizontal
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        # Cards will be added when state is set
        self._cards_layout = QHBoxLayout()
        self._cards_layout.setSpacing(4)
        self._layout.addLayout(self._cards_layout)

        self._layout.addStretch()

    def set_state(self, state: GameState) -> None:
        """Update the display from game state."""
        self._state = state

        # Create cards if needed
        while len(self._player_cards) < len(state.players):
            card = PlayerCardWidget(len(self._player_cards))
            self._player_cards.append(card)
            self._cards_layout.addWidget(card)

        # Remove extra cards if needed
        while len(self._player_cards) > len(state.players):
            card = self._player_cards.pop()
            self._cards_layout.removeWidget(card)
            card.deleteLater()

        # Update cards
        current_idx = state.global_state.current_player_idx
        starting_idx = state.global_state.starting_player_idx

        for i, (player, card) in enumerate(zip(state.players, self._player_cards)):
            card.update_from_player(
                player,
                is_current=(i == current_idx),
                is_starting=(i == starting_idx)
            )
