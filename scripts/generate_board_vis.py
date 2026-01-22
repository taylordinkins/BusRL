"""Script to generate board visualizations.

Run from the project root:
    python scripts/generate_board_vis.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import load_default_board
from data.graph_vis import visualize_board, BoardVisualizer
from core.constants import Zone, BuildingType
from core.game_state import GameState
from core.board import make_edge_id


def generate_default_board_vis():
    """Generate visualization of default board before setup."""
    print("Loading default board...")
    board = load_default_board()

    print(f"Board has {len(board.nodes)} nodes and {len(board.edges)} edges")
    print(f"Train stations: {len(board.get_train_stations())}")
    print(f"Central parks: {len(board.get_central_parks())}")

    output_path = project_root / "output" / "default_board_before_setup.png"
    output_path.parent.mkdir(exist_ok=True)

    print(f"Generating visualization -> {output_path}")
    fig = visualize_board(
        board,
        title="Bus Game - Default Board (Before Setup)",
        save_path=output_path,
        show=False,
    )
    print("Done!")
    return fig


def generate_mid_game_vis():
    """Generate visualization of board with game elements."""
    print("\nLoading default board for mid-game simulation...")
    board = load_default_board()
    state = GameState.create_initial_state(board, num_players=4)

    # 1. Place initial passengers at central parks
    print("Placing initial passengers at central parks...")
    parks = board.get_central_parks()
    for park in parks:
        p = state.passenger_manager.create_passenger(location=park.node_id)
        park.add_passenger(p.passenger_id)

    # 2. Place buildings in zones A and B
    print("Placing buildings...")
    zone_a_slots = board.get_all_empty_slots_by_zone(Zone.A)
    zone_b_slots = board.get_all_empty_slots_by_zone(Zone.B)

    building_types = [BuildingType.HOUSE, BuildingType.OFFICE, BuildingType.PUB]
    for i, (node_id, slot) in enumerate(zone_a_slots[:8]):
        slot.place_building(building_types[i % 3])
    for i, (node_id, slot) in enumerate(zone_b_slots[:4]):
        slot.place_building(building_types[(i + 1) % 3])

    # 3. Place rails for players
    print("Placing rail networks for 3 players...")

    # Player 0 (Red) - left side network
    player_0_edges = [(0, 1), (1, 5), (5, 10), (10, 18), (18, 13), (13, 14)]
    for a, b in player_0_edges:
        edge_id = make_edge_id(a, b)
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=0)

    # Player 1 (Blue) - right side network
    player_1_edges = [(3, 8), (8, 9), (9, 17), (8, 12), (12, 16), (16, 25)]
    for a, b in player_1_edges:
        edge_id = make_edge_id(a, b)
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=1)

    # Player 2 (Teal) - bottom network with some shared edges
    player_2_edges = [(27, 30), (30, 26), (26, 22), (22, 18), (27, 33), (27, 31)]
    for a, b in player_2_edges:
        edge_id = make_edge_id(a, b)
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=2)

    # Add some shared edges (multiple players)
    shared_edge = make_edge_id(14, 20)
    if shared_edge in board.edges:
        board.edges[shared_edge].add_rail(player_id=0)
        board.edges[shared_edge].add_rail(player_id=2)

    # 4. Add some passengers at train stations
    print("Adding passengers at train stations...")
    stations = board.get_train_stations()
    for station in stations:
        for _ in range(2):
            p = state.passenger_manager.create_passenger(location=station.node_id)
            station.add_passenger(p.passenger_id)

    output_path = project_root / "output" / "board_mid_game_simulation.png"
    output_path.parent.mkdir(exist_ok=True)

    print(f"Generating visualization -> {output_path}")
    fig = visualize_board(
        board,
        title="Bus Game - Mid-Game State (Simulated)",
        save_path=output_path,
        show=False,
    )
    print("Done!")
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    generate_default_board_vis()
    generate_mid_game_vis()

    print("\nVisualizations saved to output/ directory")
