"""Board widget for visualizing the game board graph."""

from __future__ import annotations

import math
from typing import Optional, Callable, Any

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QPolygonF, QMouseEvent, QPaintEvent, QResizeEvent
)

from core.board import BoardGraph, NodeId, EdgeId
from core.game_state import GameState
from core.constants import BuildingType, Zone

from gui.constants import (
    PLAYER_COLORS, ZONE_COLORS, BUILDING_COLORS, BUILDING_BORDER_COLORS,
    NODE_COLORS, EDGE_COLORS, PASSENGER_COLOR, PASSENGER_BG_COLOR,
    NODE_RADIUS, EDGE_WIDTH, RAIL_WIDTH, BUILDING_SIZE,
)


class BoardWidget(QWidget):
    """Widget for displaying and interacting with the game board.

    Signals:
        node_clicked: Emitted when a node is clicked (node_id)
        edge_clicked: Emitted when an edge is clicked (edge_id as tuple)
        node_hovered: Emitted when mouse hovers over a node (node_id or None)
        edge_hovered: Emitted when mouse hovers over an edge (edge_id or None)
    """

    node_clicked = Signal(int)
    edge_clicked = Signal(tuple)
    building_slot_clicked = Signal(int, int)  # node_id, slot_index
    node_hovered = Signal(object)  # int or None
    edge_hovered = Signal(object)  # tuple or None

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._state: Optional[GameState] = None
        self._board: Optional[BoardGraph] = None

        # Visual state
        self._node_positions: dict[NodeId, QPointF] = {}
        self._scale = 1.0
        self._offset = QPointF(0, 0)

        # Interaction state
        self._hovered_node: Optional[NodeId] = None
        self._hovered_edge: Optional[EdgeId] = None
        self._selected_nodes: set[NodeId] = set()
        self._selected_edges: set[EdgeId] = set()
        self._highlighted_nodes: set[NodeId] = set()
        self._highlighted_edges: set[EdgeId] = set()
        self._highlighted_slots: set[tuple[int, int]] = set()  # set of (node_id, slot_index)
        self._distribution_preview: dict[int, int] = {}  # node_id -> count

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

        # Set minimum size
        self.setMinimumSize(400, 300)

        # Background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#F5F5F5"))
        self.setPalette(palette)

    def set_state(self, state: GameState) -> None:
        """Update the displayed game state."""
        self._state = state
        self._board = state.board
        self._calculate_positions()
        self.update()

    def set_highlighted_nodes(self, nodes: set[NodeId]) -> None:
        """Set nodes to highlight (e.g., valid placement targets)."""
        self._highlighted_nodes = nodes
        self.update()

    def set_highlighted_edges(self, edges: set[EdgeId]) -> None:
        """Set edges to highlight (e.g., valid rail placements)."""
        self._highlighted_edges = edges
        self.update()

    def set_highlighted_slots(self, slots: set[tuple[int, int]]) -> None:
        """Set building slots to highlight."""
        self._highlighted_slots = slots
        self.update()

    def set_distribution_preview(self, distribution: dict[int, int]) -> None:
        """Set distribution counts to preview on stations."""
        self._distribution_preview = distribution
        self.update()

    def clear_highlights(self) -> None:
        """Clear all highlights."""
        self._highlighted_nodes = set()
        self._highlighted_edges = set()
        self._highlighted_slots = set()
        self._distribution_preview = {}
        self.update()

    def _calculate_positions(self) -> None:
        """Calculate screen positions for all nodes.

        The board coordinates have y=0 at the bottom, but screen coordinates
        have y=0 at the top, so we flip the y-axis.
        """
        if self._board is None:
            return

        # Get bounds of node positions
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for node in self._board.nodes.values():
            x, y = node.position
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Add generous padding for node labels and building indicators
        padding = 80

        # Board dimensions in board coordinates
        board_width = max_x - min_x
        board_height = max_y - min_y

        # Calculate scale to fit in widget with good spacing
        # Use a larger minimum scale for better visibility
        widget_width = self.width() - 2 * padding
        widget_height = self.height() - 2 * padding

        if board_width > 0 and board_height > 0:
            scale_x = widget_width / board_width
            scale_y = widget_height / board_height
            # Use the smaller scale to fit, but ensure minimum spacing
            self._scale = min(scale_x, scale_y)
            # Enforce a minimum scale of 70 pixels per board unit for readability
            self._scale = max(self._scale, 70.0)
        else:
            self._scale = 70.0  # Default pixels per board unit

        # Calculate offset to center the board
        scaled_width = board_width * self._scale
        scaled_height = board_height * self._scale

        offset_x = (self.width() - scaled_width) / 2 - min_x * self._scale
        # Flip y-axis: screen y increases downward, board y increases upward
        offset_y = (self.height() + scaled_height) / 2 + min_y * self._scale

        # Calculate screen positions
        self._node_positions.clear()
        for node_id, node in self._board.nodes.items():
            x, y = node.position
            screen_x = x * self._scale + offset_x
            # Flip y: subtract from offset instead of adding
            screen_y = offset_y - y * self._scale
            self._node_positions[node_id] = QPointF(screen_x, screen_y)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize."""
        super().resizeEvent(event)
        self._calculate_positions()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the board."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._board is None:
            # Draw placeholder text
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No game loaded")
            return

        # Draw edges first (behind nodes)
        self._draw_edges(painter)

        # Draw nodes
        self._draw_nodes(painter)

        # Draw buildings
        self._draw_buildings(painter)

        # Draw passengers
        self._draw_passengers(painter)

    def _draw_edges(self, painter: QPainter) -> None:
        """Draw all edges with rail segments."""
        if self._board is None:
            return

        for edge_id, edge in self._board.edges.items():
            node_a, node_b = edge_id
            if node_a not in self._node_positions or node_b not in self._node_positions:
                continue

            pos_a = self._node_positions[node_a]
            pos_b = self._node_positions[node_b]

            is_highlighted = edge_id in self._highlighted_edges
            is_hovered = edge_id == self._hovered_edge

            player_ids = list(edge.get_player_ids())

            if not player_ids:
                # Empty edge
                if is_highlighted or is_hovered:
                    color = QColor("#FFD700") if is_highlighted else EDGE_COLORS["hover"]
                    width = EDGE_WIDTH + (4 if is_highlighted else 2)
                    # Glow effect
                    pen = QPen(color.lighter(130), width + 2)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.setOpacity(0.5)
                    painter.drawLine(pos_a, pos_b)
                    painter.setOpacity(1.0)
                else:
                    color = EDGE_COLORS["empty"]
                    width = EDGE_WIDTH
                
                pen = QPen(color, width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(pos_a, pos_b)
            else:
                # Draw rails for each player
                num_players = len(player_ids)
                for i, player_id in enumerate(player_ids):
                    color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                    if is_hovered:
                        color = color.lighter(120)

                    # Calculate offset for parallel lines
                    offset = (i - (num_players - 1) / 2) * 4
                    offset_line = self._get_offset_line(pos_a, pos_b, offset)

                    pen = QPen(color, RAIL_WIDTH)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawLine(offset_line[0], offset_line[1])

    def _get_offset_line(
        self,
        p1: QPointF,
        p2: QPointF,
        offset: float
    ) -> tuple[QPointF, QPointF]:
        """Calculate a parallel line offset perpendicular to the original."""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            return (p1, p2)

        # Perpendicular unit vector
        px = -dy / length
        py = dx / length

        # Apply offset
        return (
            QPointF(p1.x() + px * offset, p1.y() + py * offset),
            QPointF(p2.x() + px * offset, p2.y() + py * offset)
        )

    def _draw_nodes(self, painter: QPainter) -> None:
        """Draw all nodes."""
        if self._board is None:
            return

        for node_id, node in self._board.nodes.items():
            if node_id not in self._node_positions:
                continue

            pos = self._node_positions[node_id]
            is_highlighted = node_id in self._highlighted_nodes
            is_hovered = node_id == self._hovered_node

            # Determine node color
            if node.is_train_station:
                fill_color = NODE_COLORS["train_station"]
            elif node.is_central_park:
                fill_color = NODE_COLORS["central_park"]
            else:
                fill_color = NODE_COLORS["regular"]

            if is_hovered:
                fill_color = fill_color.lighter(110)

            # Determine border color based on building slots
            if node.building_slots:
                border_color = ZONE_COLORS[node.building_slots[0].zone]
            else:
                border_color = QColor("#333333")

            # Draw highlight ring if highlighted
            if is_highlighted:
                # Brighter glow
                pen = QPen(QColor("#FFD700"), 6)
                painter.setPen(pen)
                painter.setOpacity(0.4)
                painter.drawEllipse(pos, NODE_RADIUS + 6, NODE_RADIUS + 6)
                painter.setOpacity(1.0)
                
                pen = QPen(QColor("#FFD700"), 3)
                painter.setPen(pen)
                painter.drawEllipse(pos, NODE_RADIUS + 4, NODE_RADIUS + 4)

            # Draw node
            pen = QPen(border_color, 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(pos, NODE_RADIUS, NODE_RADIUS)

            # Draw distribution preview if active
            if node_id in self._distribution_preview:
                count = self._distribution_preview[node_id]
                if count > 0:
                    # Offset badge for distribution count
                    badge_pos = QPointF(pos.x(), pos.y() + NODE_RADIUS + 25)
                    self._draw_count_badge(painter, badge_pos, f"+{count}", QColor("#00AA00"), QColor("#CCFFCC"))

            # Draw node ID
            painter.setPen(QColor("#000000"))
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            rect = QRectF(pos.x() - NODE_RADIUS, pos.y() - NODE_RADIUS,
                         NODE_RADIUS * 2, NODE_RADIUS * 2)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(node_id))

    def _draw_buildings(self, painter: QPainter) -> None:
        """Draw building indicators below nodes."""
        if self._board is None:
            return

        for node_id, node in self._board.nodes.items():
            if node_id not in self._node_positions:
                continue

            pos = self._node_positions[node_id]
            num_slots = len(node.building_slots)

            for i, slot in enumerate(node.building_slots):
                # Position below the node
                offset_x = (i - (num_slots - 1) / 2) * (BUILDING_SIZE + 4)
                bx = pos.x() + offset_x
                by = pos.y() + NODE_RADIUS + 10

                is_slot_highlighted = (node_id, i) in self._highlighted_slots

                if slot.building:
                    # Draw building marker
                    self._draw_building_marker(painter, bx, by, slot.building,
                                              slot.occupied_by_passenger_id is not None)
                else:
                    # Draw empty slot marker (zone colored square with letter)
                    size = BUILDING_SIZE + 2
                    rect = QRectF(bx - size/2, by - size/2, size, size)
                    
                    if is_slot_highlighted:
                        # Draw a shiny highlighted square
                        pen = QPen(QColor("#FFD700"), 3)
                        painter.setPen(pen)
                        painter.setBrush(QBrush(ZONE_COLORS[slot.zone].lighter(180)))
                        painter.drawRect(rect)
                    else:
                        # Draw regular zone square
                        painter.setPen(QPen(ZONE_COLORS[slot.zone], 1))
                        painter.setBrush(QBrush(ZONE_COLORS[slot.zone].lighter(150)))
                        painter.drawRect(rect)
                    
                    # Draw Zone letter
                    painter.setPen(QPen(ZONE_COLORS[slot.zone].darker(150), 1))
                    painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, slot.zone.value)

    def _draw_building_marker(
        self,
        painter: QPainter,
        x: float,
        y: float,
        building_type: BuildingType,
        is_occupied: bool
    ) -> None:
        """Draw a building marker at the given position."""
        fill_color = BUILDING_COLORS[building_type]
        border_color = BUILDING_BORDER_COLORS[building_type]

        if is_occupied:
            # Show occupied with darker color and a symbol
            fill_color = fill_color.darker(130)

        pen = QPen(border_color, 1.5)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        size = BUILDING_SIZE
        half = size / 2
        rect = QRectF(x - half, y - half, size, size)

        if building_type == BuildingType.HOUSE:
            # Square
            painter.drawRect(rect)
        elif building_type == BuildingType.OFFICE:
            # Triangle
            points = QPolygonF([
                QPointF(x, y - half),
                QPointF(x - half, y + half),
                QPointF(x + half, y + half),
            ])
            painter.drawPolygon(points)
        elif building_type == BuildingType.PUB:
            # Circle
            painter.drawEllipse(QPointF(x, y), half, half)

        # Draw "occupied" indicator if needed
        if is_occupied:
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            # Draw a thick dot or cross in the middle
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.drawEllipse(QPointF(x, y), 2.5, 2.5)

    def _draw_passengers(self, painter: QPainter) -> None:
        """Draw passenger count badges above nodes."""
        if self._board is None or self._state is None:
            return

        for node_id, node in self._board.nodes.items():
            if node_id not in self._node_positions:
                continue

            passenger_count = len(node.passenger_ids)
            if passenger_count == 0:
                continue

            pos = self._node_positions[node_id]
            # Position above the node
            badge_pos = QPointF(pos.x(), pos.y() - NODE_RADIUS - 15)
            self._draw_count_badge(painter, badge_pos, f"{passenger_count}P", PASSENGER_COLOR, PASSENGER_BG_COLOR)

    def _draw_count_badge(
        self,
        painter: QPainter,
        pos: QPointF,
        text: str,
        color: QColor,
        bg_color: QColor
    ) -> None:
        """Helper to draw a small count badge."""
        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)

        # Calculate text size
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()

        padding = 4
        badge_width = text_width + padding * 2
        badge_height = text_height + padding

        # Draw rounded rectangle
        rect = QRectF(pos.x() - badge_width / 2, pos.y() - badge_height / 2,
                     badge_width, badge_height)

        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(rect, 4, 4)

        # Draw text
        painter.setPen(color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement for hover effects."""
        pos = event.position()

        # Check for node hover
        new_hovered_node = self._get_node_at(pos)
        if new_hovered_node != self._hovered_node:
            self._hovered_node = new_hovered_node
            self.node_hovered.emit(new_hovered_node)
            self.update()

        # Check for edge hover (only if not hovering a node)
        if self._hovered_node is None:
            new_hovered_edge = self._get_edge_at(pos)
            if new_hovered_edge != self._hovered_edge:
                self._hovered_edge = new_hovered_edge
                self.edge_hovered.emit(new_hovered_edge)
                self.update()
        else:
            if self._hovered_edge is not None:
                self._hovered_edge = None
                self.edge_hovered.emit(None)
                self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse clicks."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position()

        # Check for node click
        node_id = self._get_node_at(pos)
        if node_id is not None:
            self.node_clicked.emit(node_id)
            return

        # Check for building slot click
        slot_info = self._get_slot_at(pos)
        if slot_info is not None:
            self.building_slot_clicked.emit(slot_info[0], slot_info[1])
            return

        # Check for edge click
        edge_id = self._get_edge_at(pos)
        if edge_id is not None:
            self.edge_clicked.emit(edge_id)

    def _get_node_at(self, pos: QPointF) -> Optional[NodeId]:
        """Get the node at the given screen position."""
        for node_id, node_pos in self._node_positions.items():
            dx = pos.x() - node_pos.x()
            dy = pos.y() - node_pos.y()
            if dx * dx + dy * dy <= NODE_RADIUS * NODE_RADIUS:
                return node_id
        return None

    def _get_edge_at(self, pos: QPointF) -> Optional[EdgeId]:
        """Get the edge at the given screen position."""
        if self._board is None:
            return None

        click_threshold = 10  # pixels

        for edge_id in self._board.edges.keys():
            node_a, node_b = edge_id
            if node_a not in self._node_positions or node_b not in self._node_positions:
                continue

            pos_a = self._node_positions[node_a]
            pos_b = self._node_positions[node_b]

            # Calculate distance from point to line segment
            dist = self._point_to_segment_distance(pos, pos_a, pos_b)
            if dist <= click_threshold:
                return edge_id

        return None

    def _point_to_segment_distance(
        self,
        p: QPointF,
        a: QPointF,
        b: QPointF
    ) -> float:
        """Calculate the distance from point p to line segment ab."""
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()

        # Vector from a to b
        dx = bx - ax
        dy = by - ay

        # Length squared
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            # a and b are the same point
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

        # Parameter t for the projection
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq))

        # Closest point on segment
        closest_x = ax + t * dx
        closest_y = ay + t * dy

        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def _get_slot_at(self, pos: QPointF) -> Optional[tuple[int, int]]:
        """Get the building slot (node_id, slot_index) at the given screen position."""
        if self._board is None:
            return None

        slot_radius = BUILDING_SIZE / 2 + 2

        for node_id, node in self._board.nodes.items():
            if node_id not in self._node_positions:
                continue

            n_pos = self._node_positions[node_id]
            num_slots = len(node.building_slots)

            for i in range(num_slots):
                # Calculate slot position (same logic as _draw_buildings)
                offset_x = (i - (num_slots - 1) / 2) * (BUILDING_SIZE + 4)
                bx = n_pos.x() + offset_x
                by = n_pos.y() + NODE_RADIUS + 10

                dx = pos.x() - bx
                dy = pos.y() - by
                if dx * dx + dy * dy <= slot_radius * slot_radius:
                    return (node_id, i)
        return None
