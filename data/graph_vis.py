"""Graph visualization for the Bus game board using NetworkX and matplotlib.

Provides visualization of:
- Board topology (nodes/intersections and edges/streets)
- Building slots and placed buildings (by zone and type)
- Passenger locations
- Rail segments (colored by player, multiple players per edge supported)
- Train stations and central parks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from core.board import BoardGraph, NodeId
from core.constants import Zone, BuildingType


# Color schemes
PLAYER_COLORS = [
    "#E63946",  # Red
    "#457B9D",  # Blue
    "#2A9D8F",  # Teal
    "#E9C46A",  # Yellow
    "#9B5DE5",  # Purple
]

ZONE_COLORS = {
    Zone.A: "#FF6B6B",  # Red (innermost)
    Zone.B: "#4ECDC4",  # Teal
    Zone.C: "#45B7D1",  # Blue
    Zone.D: "#96CEB4",  # Green (outermost)
}

BUILDING_MARKERS = {
    BuildingType.HOUSE: "s",   # Square
    BuildingType.OFFICE: "^",  # Triangle
    BuildingType.PUB: "o",     # Circle
}

BUILDING_COLORS = {
    BuildingType.HOUSE: "#FF9999",   # Light red
    BuildingType.OFFICE: "#99CCFF",  # Light blue
    BuildingType.PUB: "#FFCC99",     # Light orange
}


class BoardVisualizer:
    """Visualizes a Bus game board using NetworkX and matplotlib."""

    def __init__(
        self,
        board: BoardGraph,
        figsize: tuple[int, int] = (14, 12),
        node_size: int = 800,
        font_size: int = 8,
    ):
        """Initialize the visualizer.

        Args:
            board: The BoardGraph to visualize.
            figsize: Figure size as (width, height).
            node_size: Base size for nodes.
            font_size: Font size for labels.
        """
        self.board = board
        self.figsize = figsize
        self.node_size = node_size
        self.font_size = font_size
        self._graph: Optional[nx.Graph] = None

    def _build_networkx_graph(self) -> nx.Graph:
        """Convert BoardGraph to NetworkX graph."""
        G = nx.Graph()

        # Add nodes with attributes
        for node_id, node in self.board.nodes.items():
            G.add_node(
                node_id,
                pos=node.position if node.position else (node_id % 10, node_id // 10),
                is_train_station=node.is_train_station,
                is_central_park=node.is_central_park,
                building_slots=node.building_slots,
                passenger_ids=node.passenger_ids,
            )

        # Add edges with attributes
        for edge_id, edge in self.board.edges.items():
            player_ids = list(edge.get_player_ids())
            G.add_edge(
                edge_id[0],
                edge_id[1],
                rail_segments=edge.rail_segments,
                player_ids=player_ids,
            )

        return G

    def _get_node_colors(self, G: nx.Graph) -> list[str]:
        """Determine node colors based on type."""
        colors = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if node_data["is_train_station"]:
                colors.append("#FFD700")  # Gold for train stations
            elif node_data["is_central_park"]:
                colors.append("#90EE90")  # Light green for central parks
            else:
                colors.append("#FFFFFF")  # White for regular nodes
        return colors

    def _get_node_edge_colors(self, G: nx.Graph) -> list[str]:
        """Determine node border colors based on zone of first slot."""
        colors = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            slots = node_data["building_slots"]
            if slots:
                # Use color of first slot's zone
                colors.append(ZONE_COLORS[slots[0].zone])
            else:
                colors.append("#333333")  # Dark gray for nodes without slots
        return colors

    def _draw_edges(
        self,
        ax: plt.Axes,
        G: nx.Graph,
        pos: dict[NodeId, tuple[float, float]],
    ) -> None:
        """Draw edges with rail segment coloring."""
        # Group edges by player configuration
        empty_edges = []
        player_edges: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)

        for u, v, data in G.edges(data=True):
            player_ids = tuple(sorted(data["player_ids"]))
            if not player_ids:
                empty_edges.append((u, v))
            else:
                player_edges[player_ids].append((u, v))

        # Draw empty edges (gray, thin)
        if empty_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=empty_edges,
                edge_color="#CCCCCC",
                width=1.5,
                ax=ax,
            )

        # Draw player edges with colors
        for player_ids, edges in player_edges.items():
            if len(player_ids) == 1:
                # Single player - solid colored line
                color = PLAYER_COLORS[player_ids[0] % len(PLAYER_COLORS)]
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    edge_color=color,
                    width=3,
                    ax=ax,
                )
            else:
                # Multiple players - draw parallel offset lines
                for i, player_id in enumerate(player_ids):
                    color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
                    # Create offset for parallel lines
                    offset = (i - (len(player_ids) - 1) / 2) * 0.03
                    for u, v in edges:
                        self._draw_offset_edge(ax, pos, u, v, color, offset)

    def _draw_offset_edge(
        self,
        ax: plt.Axes,
        pos: dict[NodeId, tuple[float, float]],
        u: int,
        v: int,
        color: str,
        offset: float,
    ) -> None:
        """Draw an edge with perpendicular offset (for parallel lines)."""
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Calculate perpendicular offset
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        if length > 0:
            # Perpendicular unit vector
            px = -dy / length
            py = dx / length

            # Apply offset
            x1_off = x1 + px * offset
            y1_off = y1 + py * offset
            x2_off = x2 + px * offset
            y2_off = y2 + py * offset

            ax.plot(
                [x1_off, x2_off],
                [y1_off, y2_off],
                color=color,
                linewidth=2.5,
                solid_capstyle="round",
            )

    def _draw_buildings(
        self,
        ax: plt.Axes,
        G: nx.Graph,
        pos: dict[NodeId, tuple[float, float]],
    ) -> None:
        """Draw building indicators on nodes."""
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            slots = node_data["building_slots"]
            x, y = pos[node_id]

            # Draw small indicators for each building slot
            num_slots = len(slots)
            for i, slot in enumerate(slots):
                # Position indicators around the node
                angle_offset = (i - (num_slots - 1) / 2) * 0.15
                bx = x + angle_offset
                by = y - 0.25  # Below the node

                if slot.building:
                    # Placed building - use building color and marker
                    marker = BUILDING_MARKERS[slot.building]
                    color = BUILDING_COLORS[slot.building]
                    ax.scatter(
                        [bx], [by],
                        marker=marker,
                        s=150,
                        c=color,
                        edgecolors="black",
                        linewidths=1,
                        zorder=5,
                    )
                else:
                    # Empty slot - show zone color as small dot
                    ax.scatter(
                        [bx], [by],
                        marker=".",
                        s=50,
                        c=ZONE_COLORS[slot.zone],
                        zorder=4,
                    )

    def _draw_passengers(
        self,
        ax: plt.Axes,
        G: nx.Graph,
        pos: dict[NodeId, tuple[float, float]],
    ) -> None:
        """Draw passenger indicators on nodes."""
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            passenger_ids = node_data["passenger_ids"]

            if passenger_ids:
                x, y = pos[node_id]
                # Draw passenger count above the node
                count = len(passenger_ids)
                ax.annotate(
                    f"{count}P",
                    (x, y + 0.3),
                    fontsize=self.font_size,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#8B0000",  # Dark red
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#FFEEEE",
                        edgecolor="#8B0000",
                        linewidth=1,
                    ),
                    zorder=6,
                )

    def visualize(
        self,
        title: str = "Bus Game Board",
        show_buildings: bool = True,
        show_passengers: bool = True,
        show_legend: bool = True,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize the board.

        Args:
            title: Title for the figure.
            show_buildings: Whether to show building indicators.
            show_passengers: Whether to show passenger indicators.
            show_legend: Whether to show the legend.
            save_path: If provided, save the figure to this path.
            show: Whether to display the figure.

        Returns:
            The matplotlib Figure object.
        """
        # Build NetworkX graph
        G = self._build_networkx_graph()
        self._graph = G

        # Get positions from node attributes
        pos = nx.get_node_attributes(G, "pos")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Draw edges first (behind nodes)
        self._draw_edges(ax, G, pos)

        # Draw nodes
        node_colors = self._get_node_colors(G)
        node_edge_colors = self._get_node_edge_colors(G)

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            edgecolors=node_edge_colors,
            linewidths=2,
            node_size=self.node_size,
            ax=ax,
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=self.font_size,
            font_weight="bold",
            ax=ax,
        )

        # Draw buildings
        if show_buildings:
            self._draw_buildings(ax, G, pos)

        # Draw passengers
        if show_passengers:
            self._draw_passengers(ax, G, pos)

        # Draw legend
        if show_legend:
            self._draw_legend(ax)

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        # Show if requested
        if show:
            plt.show()

        return fig

    def _draw_legend(self, ax: plt.Axes) -> None:
        """Draw the legend."""
        legend_elements = []

        # Node types
        legend_elements.append(
            mpatches.Patch(facecolor="#FFD700", edgecolor="black", label="Train Station")
        )
        legend_elements.append(
            mpatches.Patch(facecolor="#90EE90", edgecolor="black", label="Central Park")
        )

        # Zone colors
        for zone, color in ZONE_COLORS.items():
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor="black", label=f"Zone {zone.value}")
            )

        # Building types
        legend_elements.append(
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=BUILDING_COLORS[BuildingType.HOUSE],
                      markersize=10, markeredgecolor="black", label="House")
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=BUILDING_COLORS[BuildingType.OFFICE],
                      markersize=10, markeredgecolor="black", label="Office")
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BUILDING_COLORS[BuildingType.PUB],
                      markersize=10, markeredgecolor="black", label="Pub")
        )

        # Player rail colors
        for i, color in enumerate(PLAYER_COLORS):
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=3, label=f"Player {i} Rail")
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
        )


def visualize_board(
    board: BoardGraph,
    title: str = "Bus Game Board",
    save_path: Optional[str | Path] = None,
    show: bool = True,
    **kwargs: Any,
) -> plt.Figure:
    """Convenience function to visualize a board.

    Args:
        board: The BoardGraph to visualize.
        title: Title for the figure.
        save_path: If provided, save the figure to this path.
        show: Whether to display the figure.
        **kwargs: Additional arguments passed to BoardVisualizer.visualize()

    Returns:
        The matplotlib Figure object.
    """
    visualizer = BoardVisualizer(board)
    return visualizer.visualize(title=title, save_path=save_path, show=show, **kwargs)


def visualize_default_board(
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Load and visualize the default Bus board.

    Args:
        save_path: If provided, save the figure to this path.
        show: Whether to display the figure.

    Returns:
        The matplotlib Figure object.
    """
    from data.loader import load_default_board

    board = load_default_board()
    return visualize_board(
        board,
        title="Default Bus Board (Before Setup)",
        save_path=save_path,
        show=show,
    )


if __name__ == "__main__":
    # When run directly, visualize the default board
    visualize_default_board(save_path="default_board_vis.png")
