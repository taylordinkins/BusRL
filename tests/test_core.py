"""Tests for the core module (constants and board)."""

import pytest

from core.constants import (
    Zone,
    BuildingType,
    Phase,
    ActionAreaType,
    MIN_PLAYERS,
    MAX_PLAYERS,
    TOTAL_ACTION_MARKERS,
    TOTAL_RAIL_SEGMENTS,
    MAX_BUSES,
    INITIAL_BUSES,
    ACTION_RESOLUTION_ORDER,
    TIME_CLOCK_ORDER,
    ZONE_PRIORITY,
)
from core.board import (
    NodeId,
    EdgeId,
    make_edge_id,
    BuildingSlot,
    NodeState,
    RailSegment,
    EdgeState,
    BoardGraph,
)
from core.player import Player
from core.components import Passenger, PassengerManager
from core.action_board import ActionSlot, ActionArea, ActionBoard
from core.game_state import GlobalState, GameState
from core.constants import (
    MIN_MARKERS_PER_ROUND,
    LINE_EXPANSION_SLOTS,
    BUSES_SLOTS,
    ACTION_RESOLUTION_ORDER,
)


# =============================================================================
# Constants Tests
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_zone_values(self):
        """Zones should be A, B, C, D from inner to outer."""
        assert Zone.A.value == "A"
        assert Zone.B.value == "B"
        assert Zone.C.value == "C"
        assert Zone.D.value == "D"
        assert len(Zone) == 4

    def test_building_type_values(self):
        """Building types should be house, office, pub."""
        assert BuildingType.HOUSE.value == "house"
        assert BuildingType.OFFICE.value == "office"
        assert BuildingType.PUB.value == "pub"
        assert len(BuildingType) == 3

    def test_phase_values(self):
        """All game phases should be defined."""
        # Setup phases
        assert Phase.SETUP_BUILDINGS.value == "setup_buildings"
        assert Phase.SETUP_RAILS_FORWARD.value == "setup_rails_forward"
        assert Phase.SETUP_RAILS_REVERSE.value == "setup_rails_reverse"
        # Main game phases
        assert Phase.CHOOSING_ACTIONS.value == "choosing_actions"
        assert Phase.RESOLVING_ACTIONS.value == "resolving_actions"
        assert Phase.CLEANUP.value == "cleanup"
        assert Phase.GAME_OVER.value == "game_over"
        assert len(Phase) == 7

    def test_action_area_type_values(self):
        """All action areas should be defined."""
        assert ActionAreaType.LINE_EXPANSION.value == "line_expansion"
        assert ActionAreaType.BUSES.value == "buses"
        assert ActionAreaType.PASSENGERS.value == "passengers"
        assert ActionAreaType.BUILDINGS.value == "buildings"
        assert ActionAreaType.TIME_CLOCK.value == "time_clock"
        assert ActionAreaType.VRROOMM.value == "vrroomm"
        assert ActionAreaType.STARTING_PLAYER.value == "starting_player"
        assert len(ActionAreaType) == 7


class TestConstants:
    """Test constant values."""

    def test_player_limits(self):
        """Player count should be 3-5."""
        assert MIN_PLAYERS == 3
        assert MAX_PLAYERS == 5

    def test_resource_limits(self):
        """Resource limits should match game rules."""
        assert TOTAL_ACTION_MARKERS == 20
        assert TOTAL_RAIL_SEGMENTS == 25
        assert MAX_BUSES == 5
        assert INITIAL_BUSES == 1

    def test_time_stones(self):
        """Should have 5 time stones by default."""
        gameState = GameState.create_initial_state(BoardGraph(), 4)
        assert gameState.global_state.time_stones_remaining == 5

    def test_action_resolution_order(self):
        """Action resolution order should be correct."""
        assert ACTION_RESOLUTION_ORDER == [
            ActionAreaType.LINE_EXPANSION,
            ActionAreaType.BUSES,
            ActionAreaType.PASSENGERS,
            ActionAreaType.BUILDINGS,
            ActionAreaType.TIME_CLOCK,
            ActionAreaType.VRROOMM,
            ActionAreaType.STARTING_PLAYER,
        ]

    def test_time_clock_order(self):
        """Time clock should cycle House -> Office -> Pub."""
        assert TIME_CLOCK_ORDER == [
            BuildingType.HOUSE,
            BuildingType.OFFICE,
            BuildingType.PUB,
        ]

    def test_zone_priority(self):
        """Zone priority should be A -> B -> C -> D (inner first)."""
        assert ZONE_PRIORITY == [Zone.A, Zone.B, Zone.C, Zone.D]


# =============================================================================
# Board Tests
# =============================================================================

class TestMakeEdgeId:
    """Test edge ID creation."""

    def test_canonical_order(self):
        """Edge IDs should always have smaller node first."""
        assert make_edge_id(1, 5) == (1, 5)
        assert make_edge_id(5, 1) == (1, 5)
        assert make_edge_id(0, 0) == (0, 0)

    def test_type(self):
        """Edge ID should be a tuple of two ints."""
        edge_id = make_edge_id(3, 7)
        assert isinstance(edge_id, tuple)
        assert len(edge_id) == 2


class TestBuildingSlot:
    """Test BuildingSlot dataclass."""

    def test_empty_slot(self):
        """New slot should be empty."""
        slot = BuildingSlot(zone=Zone.A)
        assert slot.is_empty()
        assert slot.building is None

    def test_place_building(self):
        """Should be able to place a building in empty slot."""
        slot = BuildingSlot(zone=Zone.B)
        slot.place_building(BuildingType.HOUSE)
        assert not slot.is_empty()
        assert slot.building == BuildingType.HOUSE

    def test_place_building_twice_raises(self):
        """Should raise error when placing in occupied slot."""
        slot = BuildingSlot(zone=Zone.C, building=BuildingType.PUB)
        with pytest.raises(ValueError):
            slot.place_building(BuildingType.OFFICE)

    def test_passenger_occupancy(self):
        """Should handle temporary passenger occupancy."""
        slot = BuildingSlot(zone=Zone.A)
        assert not slot.is_occupied_by_passenger()

        slot.occupy(passenger_id=10)
        assert slot.is_occupied_by_passenger()
        assert slot.occupied_by_passenger_id == 10

        with pytest.raises(ValueError):
            slot.occupy(passenger_id=11)

        slot.vacate()
        assert not slot.is_occupied_by_passenger()
        assert slot.occupied_by_passenger_id is None


class TestNodeState:
    """Test NodeState dataclass."""

    def test_default_values(self):
        """Node should have sensible defaults."""
        node = NodeState(node_id=0)
        assert node.node_id == 0
        assert node.building_slots == []
        assert node.passenger_ids == set()
        assert not node.is_train_station
        assert not node.is_central_park

    def test_get_empty_slots(self):
        """Should return only empty slots."""
        node = NodeState(
            node_id=1,
            building_slots=[
                BuildingSlot(zone=Zone.A, building=BuildingType.HOUSE),
                BuildingSlot(zone=Zone.B),
                BuildingSlot(zone=Zone.C),
            ]
        )
        empty = node.get_empty_slots()
        assert len(empty) == 2
        assert all(slot.is_empty() for slot in empty)

    def test_get_empty_slots_by_zone(self):
        """Should filter empty slots by zone."""
        node = NodeState(
            node_id=2,
            building_slots=[
                BuildingSlot(zone=Zone.A),
                BuildingSlot(zone=Zone.A, building=BuildingType.OFFICE),
                BuildingSlot(zone=Zone.B),
            ]
        )
        empty_a = node.get_empty_slots_by_zone(Zone.A)
        assert len(empty_a) == 1
        assert empty_a[0].zone == Zone.A

    def test_has_building_type(self):
        """Should detect presence of building type."""
        node = NodeState(
            node_id=3,
            building_slots=[
                BuildingSlot(zone=Zone.A, building=BuildingType.PUB),
            ]
        )
        assert node.has_building_type(BuildingType.PUB)
        assert not node.has_building_type(BuildingType.HOUSE)

    def test_passenger_management(self):
        """Should add and remove passengers."""
        node = NodeState(node_id=4)
        node.add_passenger(100)
        node.add_passenger(101)
        assert 100 in node.passenger_ids
        assert 101 in node.passenger_ids

        node.remove_passenger(100)
        assert 100 not in node.passenger_ids
        assert 101 in node.passenger_ids

    def test_remove_missing_passenger_raises(self):
        """Should raise error when removing non-existent passenger."""
        node = NodeState(node_id=5)
        with pytest.raises(KeyError):
            node.remove_passenger(999)


class TestEdgeState:
    """Test EdgeState dataclass."""

    def test_empty_edge(self):
        """New edge should have no rails."""
        edge = EdgeState(edge_id=(0, 1))
        assert edge.is_empty()
        assert edge.get_player_ids() == set()
        assert edge.endpoints == (0, 1)

    def test_add_rail(self):
        """Should add rail for a player."""
        edge = EdgeState(edge_id=(2, 3))
        edge.add_rail(player_id=0)
        assert not edge.is_empty()
        assert edge.has_player_rail(0)
        assert not edge.has_player_rail(1)

    def test_multiple_players_on_edge(self):
        """Multiple players can have rails on same edge."""
        edge = EdgeState(edge_id=(4, 5))
        edge.add_rail(player_id=0)
        edge.add_rail(player_id=1)
        assert edge.get_player_ids() == {0, 1}

    def test_duplicate_rail_raises(self):
        """Should raise error if player already has rail on edge."""
        edge = EdgeState(edge_id=(6, 7))
        edge.add_rail(player_id=0)
        with pytest.raises(ValueError):
            edge.add_rail(player_id=0)


class TestBoardGraph:
    """Test BoardGraph class."""

    @pytest.fixture
    def simple_graph(self) -> BoardGraph:
        """Create a simple test graph: 0 -- 1 -- 2"""
        graph = BoardGraph()

        # Add nodes
        graph.nodes[0] = NodeState(
            node_id=0,
            building_slots=[BuildingSlot(zone=Zone.A)],
            is_train_station=True,
        )
        graph.nodes[1] = NodeState(
            node_id=1,
            building_slots=[BuildingSlot(zone=Zone.B)],
            is_central_park=True,
        )
        graph.nodes[2] = NodeState(
            node_id=2,
            building_slots=[BuildingSlot(zone=Zone.C)],
        )

        # Add edges
        graph.edges[(0, 1)] = EdgeState(edge_id=(0, 1))
        graph.edges[(1, 2)] = EdgeState(edge_id=(1, 2))

        # Set adjacency
        graph.adjacency[0] = {1}
        graph.adjacency[1] = {0, 2}
        graph.adjacency[2] = {1}

        return graph

    def test_get_node(self, simple_graph: BoardGraph):
        """Should retrieve node by ID."""
        node = simple_graph.get_node(1)
        assert node.node_id == 1
        assert node.is_central_park

    def test_get_edge(self, simple_graph: BoardGraph):
        """Should retrieve edge by node pair (either order)."""
        edge1 = simple_graph.get_edge(0, 1)
        edge2 = simple_graph.get_edge(1, 0)
        assert edge1 is edge2
        assert edge1.edge_id == (0, 1)

    def test_get_neighbors(self, simple_graph: BoardGraph):
        """Should return adjacent nodes."""
        assert simple_graph.get_neighbors(0) == {1}
        assert simple_graph.get_neighbors(1) == {0, 2}
        assert simple_graph.get_neighbors(2) == {1}

    def test_get_train_stations(self, simple_graph: BoardGraph):
        """Should return train station nodes."""
        stations = simple_graph.get_train_stations()
        assert len(stations) == 1
        assert stations[0].node_id == 0

    def test_get_central_parks(self, simple_graph: BoardGraph):
        """Should return central park nodes."""
        parks = simple_graph.get_central_parks()
        assert len(parks) == 1
        assert parks[0].node_id == 1

    def test_get_player_edges(self, simple_graph: BoardGraph):
        """Should return edges where player has rails."""
        simple_graph.edges[(0, 1)].add_rail(player_id=0)
        player_edges = simple_graph.get_player_edges(0)
        assert len(player_edges) == 1
        assert player_edges[0].edge_id == (0, 1)

    def test_get_player_network_endpoints(self, simple_graph: BoardGraph):
        """Should identify network endpoints."""
        # Player 0 has rail on edge (0,1) only
        simple_graph.edges[(0, 1)].add_rail(player_id=0)
        endpoints = simple_graph.get_player_network_endpoints(0)
        assert endpoints == {0, 1}  # Both ends are endpoints

        # Extend to edge (1,2)
        simple_graph.edges[(1, 2)].add_rail(player_id=0)
        endpoints = simple_graph.get_player_network_endpoints(0)
        assert endpoints == {0, 2}  # Node 1 is no longer an endpoint

    def test_get_reachable_nodes(self, simple_graph: BoardGraph):
        """Should find all reachable nodes via player's network."""
        # Player 0 has rail on edge (0,1)
        simple_graph.edges[(0, 1)].add_rail(player_id=0)

        reachable_from_0 = simple_graph.get_reachable_nodes(0, start_node=0)
        assert reachable_from_0 == {0, 1}

        reachable_from_2 = simple_graph.get_reachable_nodes(0, start_node=2)
        assert reachable_from_2 == {2}  # Can't reach anywhere without rail

    def test_get_empty_edges_at_node(self, simple_graph: BoardGraph):
        """Should return empty edges adjacent to a node."""
        empty = simple_graph.get_empty_edges_at_node(1)
        assert len(empty) == 2

        # Add rail to one edge
        simple_graph.edges[(0, 1)].add_rail(player_id=0)
        empty = simple_graph.get_empty_edges_at_node(1)
        assert len(empty) == 1
        assert empty[0].edge_id == (1, 2)


# =============================================================================
# Player Tests
# =============================================================================

class TestPlayer:
    """Test Player dataclass."""

    def test_default_values(self):
        """New player should have correct starting resources."""
        player = Player(player_id=0)
        assert player.player_id == 0
        assert player.action_markers_remaining == TOTAL_ACTION_MARKERS
        assert player.rail_segments_remaining == TOTAL_RAIL_SEGMENTS
        assert player.buses == INITIAL_BUSES
        assert player.score == 0
        assert player.time_stones == 0
        assert not player.has_passed
        assert player.markers_placed_this_round == 0

    def test_place_marker(self):
        """Should decrement markers and track placement."""
        player = Player(player_id=0)
        initial = player.action_markers_remaining
        player.place_marker()
        assert player.action_markers_remaining == initial - 1
        assert player.markers_placed_this_round == 1

    def test_place_marker_when_passed_raises(self):
        """Should raise error if player has passed."""
        player = Player(player_id=0)
        player.pass_turn()
        with pytest.raises(ValueError):
            player.place_marker()

    def test_place_marker_when_none_remaining_raises(self):
        """Should raise error if no markers remaining."""
        player = Player(player_id=0, action_markers_remaining=0)
        with pytest.raises(ValueError):
            player.place_marker()

    def test_place_rail(self):
        """Should decrement rail segments."""
        player = Player(player_id=0)
        initial = player.rail_segments_remaining
        player.place_rail()
        assert player.rail_segments_remaining == initial - 1

    def test_place_rail_when_none_remaining_raises(self):
        """Should raise error if no rails remaining."""
        player = Player(player_id=0, rail_segments_remaining=0)
        with pytest.raises(ValueError):
            player.place_rail()

    def test_gain_bus(self):
        """Should increment bus count."""
        player = Player(player_id=0)
        assert player.buses == INITIAL_BUSES
        player.gain_bus()
        assert player.buses == INITIAL_BUSES + 1

    def test_gain_bus_at_max_raises(self):
        """Should raise error if already at max buses."""
        player = Player(player_id=0, buses=MAX_BUSES)
        with pytest.raises(ValueError):
            player.gain_bus()

    def test_add_score(self):
        """Should add points to score."""
        player = Player(player_id=0)
        player.add_score(5)
        assert player.score == 5
        player.add_score(-2)
        assert player.score == 3

    def test_take_time_stone(self):
        """Should increment time stones."""
        player = Player(player_id=0)
        player.take_time_stone()
        assert player.time_stones == 1

    def test_get_final_score(self):
        """Final score should subtract time stones."""
        player = Player(player_id=0, score=10, time_stones=3)
        assert player.get_final_score() == 7

    def test_pass_turn(self):
        """Should mark player as passed."""
        player = Player(player_id=0)
        assert not player.has_passed
        player.pass_turn()
        assert player.has_passed

    def test_pass_turn_twice_raises(self):
        """Should raise error if already passed."""
        player = Player(player_id=0)
        player.pass_turn()
        with pytest.raises(ValueError):
            player.pass_turn()

    def test_reset_for_new_round(self):
        """Should reset per-round state."""
        player = Player(player_id=0)
        player.place_marker()
        player.place_marker()
        player.pass_turn()

        player.reset_for_new_round()
        assert not player.has_passed
        assert player.markers_placed_this_round == 0
        # Markers should NOT be restored
        assert player.action_markers_remaining == TOTAL_ACTION_MARKERS - 2

    def test_has_resources(self):
        """Should check if player has markers remaining."""
        player = Player(player_id=0)
        assert player.has_resources()
        player.action_markers_remaining = 0
        assert not player.has_resources()


# =============================================================================
# Player + Board Integration Tests
# =============================================================================

class TestPlayerBoardIntegration:
    """Test player interactions with the board."""

    @pytest.fixture
    def game_setup(self) -> tuple[BoardGraph, list[Player]]:
        """Create a simple board and players for testing."""
        # Create board: 0 -- 1 -- 2 -- 3
        graph = BoardGraph()
        for i in range(4):
            graph.nodes[i] = NodeState(
                node_id=i,
                building_slots=[BuildingSlot(zone=Zone.A)],
            )
        for i in range(3):
            edge_id = make_edge_id(i, i + 1)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(i, set()).add(i + 1)
            graph.adjacency.setdefault(i + 1, set()).add(i)

        # Create 3 players
        players = [Player(player_id=i) for i in range(3)]
        return graph, players

    def test_player_builds_rail_network(self, game_setup):
        """Player can build rails and track resources."""
        graph, players = game_setup
        player = players[0]

        # Player places rail on edge (0, 1)
        edge = graph.get_edge(0, 1)
        assert edge.is_empty()
        assert player.can_place_rail()

        player.place_rail()
        edge.add_rail(player.player_id)

        assert not edge.is_empty()
        assert edge.has_player_rail(player.player_id)
        assert player.rail_segments_remaining == TOTAL_RAIL_SEGMENTS - 1

    def test_player_network_grows(self, game_setup):
        """Player network endpoints update as rails are placed."""
        graph, players = game_setup
        player = players[0]

        # Place first rail
        graph.edges[(0, 1)].add_rail(player.player_id)
        player.place_rail()
        endpoints = graph.get_player_network_endpoints(player.player_id)
        assert endpoints == {0, 1}

        # Extend network
        graph.edges[(1, 2)].add_rail(player.player_id)
        player.place_rail()
        endpoints = graph.get_player_network_endpoints(player.player_id)
        assert endpoints == {0, 2}

    def test_multiple_players_share_edge(self, game_setup):
        """Multiple players can have rails on the same edge."""
        graph, players = game_setup
        edge = graph.get_edge(1, 2)

        for player in players[:2]:
            player.place_rail()
            edge.add_rail(player.player_id)

        assert edge.get_player_ids() == {0, 1}
        assert not edge.has_player_rail(2)

    def test_player_reachability_on_network(self, game_setup):
        """Player can only reach nodes through their own rails."""
        graph, players = game_setup

        # Player 0 builds rails on edges (0,1) and (1,2)
        graph.edges[(0, 1)].add_rail(0)
        graph.edges[(1, 2)].add_rail(0)

        # Player 1 builds rail on edge (2,3)
        graph.edges[(2, 3)].add_rail(1)

        # Player 0 can reach nodes 0, 1, 2 from node 0
        reachable = graph.get_reachable_nodes(0, start_node=0)
        assert reachable == {0, 1, 2}

        # Player 0 cannot reach node 3 (player 1's rail)
        assert 3 not in reachable

        # Player 1 can only reach nodes 2, 3 from node 3
        reachable = graph.get_reachable_nodes(1, start_node=3)
        assert reachable == {2, 3}


# =============================================================================
# Components Tests
# =============================================================================

class TestPassenger:
    """Test Passenger dataclass."""

    def test_creation(self):
        """Passenger should store ID and location."""
        passenger = Passenger(passenger_id=0, location=5)
        assert passenger.passenger_id == 0
        assert passenger.location == 5

    def test_move_to(self):
        """Passenger can move to a new location."""
        passenger = Passenger(passenger_id=1, location=3)
        passenger.move_to(7)
        assert passenger.location == 7


class TestPassengerManager:
    """Test PassengerManager class."""

    def test_create_passenger(self):
        """Should create passengers with unique IDs."""
        manager = PassengerManager()
        p1 = manager.create_passenger(location=0)
        p2 = manager.create_passenger(location=1)

        assert p1.passenger_id == 0
        assert p2.passenger_id == 1
        assert p1.location == 0
        assert p2.location == 1
        assert manager.count() == 2

    def test_get_passenger(self):
        """Should retrieve passenger by ID."""
        manager = PassengerManager()
        p1 = manager.create_passenger(location=5)

        retrieved = manager.get_passenger(p1.passenger_id)
        assert retrieved is p1

        assert manager.get_passenger(999) is None

    def test_get_passengers_at(self):
        """Should find all passengers at a location."""
        manager = PassengerManager()
        manager.create_passenger(location=3)
        manager.create_passenger(location=3)
        manager.create_passenger(location=5)

        at_3 = manager.get_passengers_at(3)
        assert len(at_3) == 2
        assert all(p.location == 3 for p in at_3)

        at_5 = manager.get_passengers_at(5)
        assert len(at_5) == 1

        at_99 = manager.get_passengers_at(99)
        assert len(at_99) == 0

    def test_move_passenger(self):
        """Should move passenger to new location."""
        manager = PassengerManager()
        p = manager.create_passenger(location=0)

        manager.move_passenger(p.passenger_id, new_location=10)
        assert p.location == 10

    def test_move_nonexistent_passenger_raises(self):
        """Should raise error for invalid passenger ID."""
        manager = PassengerManager()
        with pytest.raises(KeyError):
            manager.move_passenger(999, new_location=5)

    def test_get_all_locations(self):
        """Should return mapping of locations to passenger IDs."""
        manager = PassengerManager()
        p1 = manager.create_passenger(location=1)
        p2 = manager.create_passenger(location=1)
        p3 = manager.create_passenger(location=2)

        locations = manager.get_all_locations()
        assert set(locations[1]) == {p1.passenger_id, p2.passenger_id}
        assert locations[2] == [p3.passenger_id]


# =============================================================================
# Passenger + Board Integration Tests
# =============================================================================

class TestPassengerBoardIntegration:
    """Test passenger interactions with the board."""

    @pytest.fixture
    def board_with_passengers(self) -> tuple[BoardGraph, PassengerManager]:
        """Create a board with nodes and a passenger manager."""
        graph = BoardGraph()

        # Create nodes: train station, central park, regular node with building
        graph.nodes[0] = NodeState(
            node_id=0,
            is_train_station=True,
        )
        graph.nodes[1] = NodeState(
            node_id=1,
            is_central_park=True,
            building_slots=[BuildingSlot(zone=Zone.A), BuildingSlot(zone=Zone.A)],
        )
        graph.nodes[2] = NodeState(
            node_id=2,
            building_slots=[BuildingSlot(zone=Zone.B, building=BuildingType.HOUSE)],
        )

        # Connect nodes
        for i in range(2):
            edge_id = make_edge_id(i, i + 1)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(i, set()).add(i + 1)
            graph.adjacency.setdefault(i + 1, set()).add(i)

        manager = PassengerManager()
        return graph, manager

    def test_spawn_passengers_at_station(self, board_with_passengers):
        """Passengers can be spawned at train stations."""
        graph, manager = board_with_passengers
        stations = graph.get_train_stations()
        assert len(stations) == 1

        station = stations[0]
        p1 = manager.create_passenger(location=station.node_id)
        station.add_passenger(p1.passenger_id)

        assert p1.passenger_id in station.passenger_ids
        assert len(manager.get_passengers_at(station.node_id)) == 1

    def test_spawn_passengers_at_central_park(self, board_with_passengers):
        """Initial passengers start at central parks."""
        graph, manager = board_with_passengers
        parks = graph.get_central_parks()
        assert len(parks) == 1

        park = parks[0]
        p = manager.create_passenger(location=park.node_id)
        park.add_passenger(p.passenger_id)

        assert p.passenger_id in park.passenger_ids

    def test_move_passenger_along_network(self, board_with_passengers):
        """Passenger moves via player rail network to destination."""
        graph, manager = board_with_passengers

        # Player 0 builds rails connecting all nodes
        graph.edges[(0, 1)].add_rail(0)
        graph.edges[(1, 2)].add_rail(0)

        # Spawn passenger at train station (node 0)
        station = graph.get_node(0)
        p = manager.create_passenger(location=0)
        station.add_passenger(p.passenger_id)

        # Verify passenger can reach node 2 via player 0's network
        reachable = graph.get_reachable_nodes(0, start_node=0)
        assert 2 in reachable

        # Move passenger to node 2 (which has a HOUSE building)
        destination = graph.get_node(2)
        assert destination.has_building_type(BuildingType.HOUSE)

        station.remove_passenger(p.passenger_id)
        manager.move_passenger(p.passenger_id, new_location=2)
        destination.add_passenger(p.passenger_id)

        assert p.location == 2
        assert p.passenger_id in destination.passenger_ids
        assert p.passenger_id not in station.passenger_ids

    def test_passenger_delivery_to_matching_building(self, board_with_passengers):
        """Passenger delivered to building matching time clock type."""
        graph, manager = board_with_passengers

        # Node 2 has a HOUSE building
        destination = graph.get_node(2)
        assert destination.has_building_type(BuildingType.HOUSE)

        # Create and deliver passenger
        p = manager.create_passenger(location=2)
        destination.add_passenger(p.passenger_id)

        # Verify passenger is at node with matching building
        passengers_at_dest = manager.get_passengers_at(2)
        assert len(passengers_at_dest) == 1
        assert destination.get_buildings_of_type(BuildingType.HOUSE)


# =============================================================================
# Action Board Tests
# =============================================================================

class TestActionSlot:
    """Test ActionSlot dataclass."""

    def test_empty_slot(self):
        """New slot should be empty."""
        slot = ActionSlot(label="A")
        assert slot.is_empty()
        assert slot.player_id is None
        assert slot.placement_order is None

    def test_place_marker(self):
        """Should place marker in empty slot."""
        slot = ActionSlot(label="B")
        slot.place_marker(player_id=0, placement_order=5)

        assert not slot.is_empty()
        assert slot.player_id == 0
        assert slot.placement_order == 5

    def test_place_marker_occupied_raises(self):
        """Should raise error when placing in occupied slot."""
        slot = ActionSlot(label="C", player_id=1, placement_order=0)
        with pytest.raises(ValueError):
            slot.place_marker(player_id=2, placement_order=1)

    def test_clear(self):
        """Should remove marker from slot."""
        slot = ActionSlot(label="D", player_id=0, placement_order=3)
        slot.clear()

        assert slot.is_empty()
        assert slot.player_id is None
        assert slot.placement_order is None


class TestActionArea:
    """Test ActionArea dataclass."""

    def test_initialization(self):
        """Should initialize with correct number of slots (A, B, C, D, E, F)."""
        area = ActionArea(area_type=ActionAreaType.LINE_EXPANSION, max_slots=6)
        assert len(area.slots) == 6
        # Slots are stored by label
        assert "A" in area.slots
        assert "F" in area.slots

    def test_single_slot_area(self):
        """Single slot areas should have label A."""
        area = ActionArea(area_type=ActionAreaType.BUSES, max_slots=1)
        assert len(area.slots) == 1
        assert "A" in area.slots

    def test_get_next_available_slot(self):
        """Should return next empty slot in placement order (A first)."""
        area = ActionArea(area_type=ActionAreaType.VRROOMM, max_slots=6)

        # First available is A (placement order is always A, B, C, ...)
        slot = area.get_next_available_slot()
        assert slot is not None
        assert slot.label == "A"

        # Fill A, next is B
        slot.place_marker(player_id=0, placement_order=0)
        next_slot = area.get_next_available_slot()
        assert next_slot is not None
        assert next_slot.label == "B"

    def test_is_full(self):
        """Should detect when all slots are occupied."""
        area = ActionArea(area_type=ActionAreaType.BUSES, max_slots=1)
        assert not area.is_full()

        area.slots["A"].place_marker(player_id=0, placement_order=0)
        assert area.is_full()

    def test_resolution_order_reversed_layout(self):
        """LINE_EXPANSION resolves F, E, D, C, B, A (left to right)."""
        area = ActionArea(area_type=ActionAreaType.LINE_EXPANSION, max_slots=6)

        # Place markers in A, B, C
        area.slots["A"].place_marker(player_id=0, placement_order=0)
        area.slots["B"].place_marker(player_id=1, placement_order=1)
        area.slots["C"].place_marker(player_id=2, placement_order=2)

        # Resolution order for LINE_EXPANSION: F, E, D, C, B, A
        resolved = area.get_occupied_slots_in_resolution_order()
        assert len(resolved) == 3
        # Should be C, B, A (left to right in reversed layout)
        assert resolved[0].label == "C"
        assert resolved[1].label == "B"
        assert resolved[2].label == "A"

    def test_resolution_order_normal_layout(self):
        """PASSENGERS resolves A, B, C, D, E, F (left to right)."""
        area = ActionArea(area_type=ActionAreaType.PASSENGERS, max_slots=6)

        # Place markers in A, B, C
        area.slots["A"].place_marker(player_id=0, placement_order=0)
        area.slots["B"].place_marker(player_id=1, placement_order=1)
        area.slots["C"].place_marker(player_id=2, placement_order=2)

        # Resolution order for PASSENGERS: A, B, C, D, E, F
        resolved = area.get_occupied_slots_in_resolution_order()
        assert len(resolved) == 3
        # Should be A, B, C (left to right in normal layout)
        assert resolved[0].label == "A"
        assert resolved[1].label == "B"
        assert resolved[2].label == "C"

    def test_get_player_slots(self):
        """Should return all slots for a specific player."""
        area = ActionArea(area_type=ActionAreaType.BUILDINGS, max_slots=6)
        area.slots["A"].place_marker(player_id=0, placement_order=0)
        area.slots["B"].place_marker(player_id=1, placement_order=1)
        area.slots["C"].place_marker(player_id=0, placement_order=2)

        player_0_slots = area.get_player_slots(0)
        assert len(player_0_slots) == 2

        player_1_slots = area.get_player_slots(1)
        assert len(player_1_slots) == 1

    def test_clear_all(self):
        """Should remove all markers."""
        area = ActionArea(area_type=ActionAreaType.TIME_CLOCK, max_slots=1)
        area.slots["A"].place_marker(player_id=0, placement_order=0)

        area.clear_all()
        assert all(slot.is_empty() for slot in area.slots.values())


class TestActionBoard:
    """Test ActionBoard class."""

    def test_initialization(self):
        """Should create all 7 action areas with correct slot counts."""
        board = ActionBoard()

        assert len(board.areas) == 7
        assert board.get_area(ActionAreaType.LINE_EXPANSION).max_slots == LINE_EXPANSION_SLOTS
        assert board.get_area(ActionAreaType.BUSES).max_slots == BUSES_SLOTS
        assert board.placement_counter == 0

    def test_place_marker(self):
        """Should place marker in slot A first, then B, etc."""
        board = ActionBoard()

        # Placement is always A first, regardless of physical layout
        slot = board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        assert slot.label == "A"
        assert slot.player_id == 0
        assert slot.placement_order == 0
        assert board.placement_counter == 1

        slot2 = board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=1)
        assert slot2.label == "B"
        assert slot2.placement_order == 1
        assert board.placement_counter == 2

    def test_place_marker_full_area_raises(self):
        """Should raise error when area is full."""
        board = ActionBoard()

        # Fill the single-slot BUSES area
        board.place_marker(ActionAreaType.BUSES, player_id=0)

        with pytest.raises(ValueError):
            board.place_marker(ActionAreaType.BUSES, player_id=1)

    def test_can_place_marker(self):
        """Should check if area has space."""
        board = ActionBoard()

        assert board.can_place_marker(ActionAreaType.BUSES)
        board.place_marker(ActionAreaType.BUSES, player_id=0)
        assert not board.can_place_marker(ActionAreaType.BUSES)

    def test_get_available_areas(self):
        """Should return areas with empty slots."""
        board = ActionBoard()

        available = board.get_available_areas()
        assert len(available) == 7  # All areas initially

        # Fill BUSES (single slot)
        board.place_marker(ActionAreaType.BUSES, player_id=0)
        available = board.get_available_areas()
        assert ActionAreaType.BUSES not in available
        assert len(available) == 6

    def test_get_markers_to_resolve_reversed_layout(self):
        """LINE_EXPANSION resolves in physical left-to-right order (F, E, D, C, B, A)."""
        board = ActionBoard()

        # Place markers (goes to A, B, C in placement order)
        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)  # A
        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=1)  # B
        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=2)  # C

        markers = board.get_markers_to_resolve(ActionAreaType.LINE_EXPANSION)
        assert len(markers) == 3
        # Resolution is left-to-right: C, B, A (for reversed layout)
        assert markers[0].label == "C"
        assert markers[1].label == "B"
        assert markers[2].label == "A"

    def test_get_markers_to_resolve_normal_layout(self):
        """PASSENGERS resolves in physical left-to-right order (A, B, C, D, E, F)."""
        board = ActionBoard()

        # Place markers (goes to A, B, C in placement order)
        board.place_marker(ActionAreaType.PASSENGERS, player_id=0)  # A
        board.place_marker(ActionAreaType.PASSENGERS, player_id=1)  # B
        board.place_marker(ActionAreaType.PASSENGERS, player_id=2)  # C

        markers = board.get_markers_to_resolve(ActionAreaType.PASSENGERS)
        assert len(markers) == 3
        # Resolution is left-to-right: A, B, C (for normal layout)
        assert markers[0].label == "A"
        assert markers[1].label == "B"
        assert markers[2].label == "C"

    def test_get_all_player_markers(self):
        """Should return all markers for a player across all areas."""
        board = ActionBoard()

        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        board.place_marker(ActionAreaType.BUSES, player_id=1)
        board.place_marker(ActionAreaType.VRROOMM, player_id=0)
        board.place_marker(ActionAreaType.PASSENGERS, player_id=0)

        player_0_markers = board.get_all_player_markers(0)
        assert len(player_0_markers) == 3

        areas = [area_type for area_type, _ in player_0_markers]
        assert ActionAreaType.LINE_EXPANSION in areas
        assert ActionAreaType.VRROOMM in areas
        assert ActionAreaType.PASSENGERS in areas

    def test_count_total_markers(self):
        """Should count all markers on the board."""
        board = ActionBoard()

        assert board.count_total_markers() == 0

        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        board.place_marker(ActionAreaType.BUSES, player_id=1)
        board.place_marker(ActionAreaType.VRROOMM, player_id=2)

        assert board.count_total_markers() == 3

    def test_clear_all(self):
        """Should remove all markers and reset counter."""
        board = ActionBoard()

        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        board.place_marker(ActionAreaType.BUSES, player_id=1)

        board.clear_all()

        assert board.count_total_markers() == 0
        assert board.placement_counter == 0
        assert board.can_place_marker(ActionAreaType.BUSES)

    def test_get_resolution_order(self):
        """Should return correct resolution order."""
        board = ActionBoard()
        order = board.get_resolution_order()

        assert order == ACTION_RESOLUTION_ORDER
        assert order[0] == ActionAreaType.LINE_EXPANSION
        assert order[-1] == ActionAreaType.STARTING_PLAYER


# =============================================================================
# Action Board + Player Integration Tests
# =============================================================================

class TestActionBoardPlayerIntegration:
    """Test action board interactions with players."""

    def test_player_places_markers(self):
        """Player resources track marker placement."""
        board = ActionBoard()
        player = Player(player_id=0)

        # Player places marker
        player.place_marker()
        board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=player.player_id)

        assert player.markers_placed_this_round == 1
        assert player.action_markers_remaining == TOTAL_ACTION_MARKERS - 1
        assert board.count_total_markers() == 1

    def test_multiple_players_choosing_actions(self):
        """Multiple players can place markers in round-robin."""
        board = ActionBoard()
        players = [Player(player_id=i) for i in range(3)]

        # Simulate a choosing actions round
        placements = [
            (0, ActionAreaType.LINE_EXPANSION),
            (1, ActionAreaType.BUSES),
            (2, ActionAreaType.VRROOMM),
            (0, ActionAreaType.PASSENGERS),
            (1, ActionAreaType.BUILDINGS),
            (2, ActionAreaType.LINE_EXPANSION),
        ]

        for player_id, area_type in placements:
            players[player_id].place_marker()
            board.place_marker(area_type, player_id)

        assert board.count_total_markers() == 6
        assert players[0].markers_placed_this_round == 2
        assert players[1].markers_placed_this_round == 2
        assert players[2].markers_placed_this_round == 2

    def test_cleanup_phase(self):
        """Cleanup resets board but not player resources."""
        board = ActionBoard()
        player = Player(player_id=0)

        # Place markers
        for _ in range(3):
            player.place_marker()
            board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)

        markers_used = TOTAL_ACTION_MARKERS - player.action_markers_remaining

        # Cleanup
        board.clear_all()
        player.reset_for_new_round()

        # Board is cleared
        assert board.count_total_markers() == 0

        # Player markers are NOT restored
        assert player.action_markers_remaining == TOTAL_ACTION_MARKERS - markers_used
        assert player.markers_placed_this_round == 0


# =============================================================================
# Game State Tests
# =============================================================================

class TestGlobalState:
    """Test GlobalState dataclass."""

    def test_default_values(self):
        """GlobalState should have correct defaults."""
        state = GlobalState()
        assert state.round_number == 1
        assert state.current_player_idx == 0
        assert state.starting_player_idx == 0
        assert state.time_clock_position == BuildingType.HOUSE
        # Default is 5 for >= 4 players unless specified otherwise
        # (GlobalState init doesn't know player count, assumes default 5)
        assert state.time_stones_remaining == 5
        assert state.current_resolution_area_idx == 0
        assert state.current_resolution_slot_idx == 0
        assert not state.game_ended

    def test_advance_time_clock(self):
        """Time clock should cycle House -> Office -> Pub -> House."""
        state = GlobalState()

        assert state.time_clock_position == BuildingType.HOUSE
        state.advance_time_clock()
        assert state.time_clock_position == BuildingType.OFFICE
        state.advance_time_clock()
        assert state.time_clock_position == BuildingType.PUB
        state.advance_time_clock()
        assert state.time_clock_position == BuildingType.HOUSE  # Wraps around

    def test_take_time_stone(self):
        """Should decrement time stones and return success."""
        state = GlobalState()
        initial = state.time_stones_remaining

        assert state.take_time_stone()
        assert state.time_stones_remaining == initial - 1

        # Take all remaining
        while state.time_stones_remaining > 0:
            state.take_time_stone()

        # Can't take when none remaining
        assert not state.take_time_stone()
        assert state.time_stones_remaining == 0

    def test_resolution_tracking(self):
        """Should track resolution progress through areas."""
        state = GlobalState()

        # Start at first area
        area = state.get_current_resolution_area()
        assert area == ActionAreaType.LINE_EXPANSION

        # Advance through areas
        state.advance_resolution_area()
        assert state.get_current_resolution_area() == ActionAreaType.BUSES
        assert state.current_resolution_slot_idx == 0

        # Track slot within area
        state.advance_resolution_slot()
        assert state.current_resolution_slot_idx == 1

        # Advancing area resets slot
        state.advance_resolution_area()
        assert state.current_resolution_slot_idx == 0

    def test_reset_for_new_round(self):
        """Should reset resolution tracking and increment round."""
        state = GlobalState()
        state.current_resolution_area_idx = 5
        state.current_resolution_slot_idx = 3

        state.reset_for_new_round()

        assert state.round_number == 2
        assert state.current_resolution_area_idx == 0
        assert state.current_resolution_slot_idx == 0


class TestGameState:
    """Test GameState class."""

    @pytest.fixture
    def simple_board(self) -> BoardGraph:
        """Create a simple test board."""
        graph = BoardGraph()

        # Create 4 nodes: train station, central park, and 2 regular
        graph.nodes[0] = NodeState(
            node_id=0,
            is_train_station=True,
            position=(0, 0),
        )
        graph.nodes[1] = NodeState(
            node_id=1,
            is_central_park=True,
            building_slots=[BuildingSlot(zone=Zone.A)],
            position=(1, 0),
        )
        graph.nodes[2] = NodeState(
            node_id=2,
            building_slots=[BuildingSlot(zone=Zone.B)],
            position=(2, 0),
        )
        graph.nodes[3] = NodeState(
            node_id=3,
            building_slots=[BuildingSlot(zone=Zone.C)],
            position=(3, 0),
        )

        # Connect in a line: 0 -- 1 -- 2 -- 3
        for i in range(3):
            edge_id = make_edge_id(i, i + 1)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(i, set()).add(i + 1)
            graph.adjacency.setdefault(i + 1, set()).add(i)

        return graph

    def test_create_initial_state(self, simple_board: BoardGraph):
        """Should create valid initial game state."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        assert state.board is simple_board
        assert len(state.players) == 3
        assert state.phase == Phase.SETUP_BUILDINGS
        assert state.global_state.round_number == 1
        assert state.passenger_manager.count() == 0
        assert state.action_board.count_total_markers() == 0

    def test_create_initial_state_validates_player_count(self, simple_board: BoardGraph):
        """Should reject invalid player counts."""
        with pytest.raises(ValueError):
            GameState.create_initial_state(simple_board, num_players=2)
        with pytest.raises(ValueError):
            GameState.create_initial_state(simple_board, num_players=6)

    def test_time_stone_count_based_on_players(self, simple_board: BoardGraph):
        """Should set correct number of time stones based on player count."""
        # 3 Players -> 4 Time Stones
        state_3p = GameState.create_initial_state(simple_board, num_players=3)
        assert state_3p.global_state.time_stones_remaining == 4

        # 4 Players -> 5 Time Stones
        state_4p = GameState.create_initial_state(simple_board, num_players=4)
        assert state_4p.global_state.time_stones_remaining == 5

        # 5 Players -> 5 Time Stones
        state_5p = GameState.create_initial_state(simple_board, num_players=5)
        assert state_5p.global_state.time_stones_remaining == 5

    def test_player_access(self, simple_board: BoardGraph):
        """Should provide access to players."""
        state = GameState.create_initial_state(simple_board, num_players=4)

        # Get current player
        current = state.get_current_player()
        assert current.player_id == 0

        # Get player by ID
        player2 = state.get_player(2)
        assert player2.player_id == 2

        # Invalid ID
        with pytest.raises(ValueError):
            state.get_player(10)

    def test_advance_current_player(self, simple_board: BoardGraph):
        """Should cycle through players."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        assert state.global_state.current_player_idx == 0
        state.advance_current_player()
        assert state.global_state.current_player_idx == 1
        state.advance_current_player()
        assert state.global_state.current_player_idx == 2
        state.advance_current_player()
        assert state.global_state.current_player_idx == 0  # Wraps

    def test_set_starting_player(self, simple_board: BoardGraph):
        """Should set starting player for next round."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        state.set_starting_player(2)
        assert state.global_state.starting_player_idx == 2
        assert state.get_starting_player().player_id == 2

        with pytest.raises(ValueError):
            state.set_starting_player(5)

    def test_phase_management(self, simple_board: BoardGraph):
        """Should track and transition phases."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        assert state.is_setup_phase()
        assert not state.is_game_over()

        state.set_phase(Phase.CHOOSING_ACTIONS)
        assert not state.is_setup_phase()

        state.set_phase(Phase.GAME_OVER)
        assert state.is_game_over()

    def test_start_new_round(self, simple_board: BoardGraph):
        """Should reset state for a new round."""
        state = GameState.create_initial_state(simple_board, num_players=3)
        state.set_phase(Phase.CHOOSING_ACTIONS)

        # Simulate some activity
        state.players[0].place_marker()
        state.players[0].pass_turn()
        state.action_board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        state.set_starting_player(1)
        state.global_state.current_player_idx = 2

        # Start new round
        state.start_new_round()

        assert state.phase == Phase.CHOOSING_ACTIONS
        assert not state.players[0].has_passed
        assert state.players[0].markers_placed_this_round == 0
        assert state.action_board.count_total_markers() == 0
        assert state.global_state.current_player_idx == 1  # Set to starting player
        assert state.global_state.round_number == 2

    def test_all_players_passed(self, simple_board: BoardGraph):
        """Should detect when all players have passed."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        assert not state.all_players_passed()
        assert len(state.get_active_players()) == 3

        state.players[0].pass_turn()
        assert not state.all_players_passed()
        assert len(state.get_active_players()) == 2

        state.players[1].pass_turn()
        state.players[2].pass_turn()
        assert state.all_players_passed()
        assert len(state.get_active_players()) == 0

    def test_clone(self, simple_board: BoardGraph):
        """Should create independent deep copy."""
        state = GameState.create_initial_state(simple_board, num_players=3)
        state.players[0].add_score(10)
        state.passenger_manager.create_passenger(location=1)

        clone = state.clone()

        # Verify copy has same values
        assert clone.players[0].score == 10
        assert clone.passenger_manager.count() == 1

        # Modify original
        state.players[0].add_score(5)
        state.passenger_manager.create_passenger(location=2)

        # Clone should be unaffected
        assert clone.players[0].score == 10
        assert clone.passenger_manager.count() == 1

    def test_to_dict(self, simple_board: BoardGraph):
        """Should serialize state to dictionary."""
        state = GameState.create_initial_state(simple_board, num_players=3)
        state.players[0].add_score(5)
        state.action_board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)

        data = state.to_dict()

        assert data["phase"] == "setup_buildings"
        assert data["global_state"]["round_number"] == 1
        assert len(data["players"]) == 3
        assert data["players"][0]["score"] == 5
        assert data["action_board"]["placement_counter"] == 1

    def test_state_hash(self, simple_board: BoardGraph):
        """Should generate consistent state hash."""
        state1 = GameState.create_initial_state(simple_board, num_players=3)
        state2 = GameState.create_initial_state(simple_board, num_players=3)

        # Same state should have same hash
        assert state1.state_hash() == state2.state_hash()

        # Different state should have different hash
        state2.players[0].add_score(1)
        assert state1.state_hash() != state2.state_hash()

    def test_validate(self, simple_board: BoardGraph):
        """Should detect invalid states."""
        state = GameState.create_initial_state(simple_board, num_players=3)

        # Valid state
        errors = state.validate()
        assert len(errors) == 0

        # Invalid current player index
        state.global_state.current_player_idx = 10
        errors = state.validate()
        assert any("current_player_idx" in e for e in errors)

    def test_str_representation(self, simple_board: BoardGraph):
        """Should have readable string representation."""
        state = GameState.create_initial_state(simple_board, num_players=3)
        s = str(state)

        assert "GameState" in s
        assert "phase=setup_buildings" in s
        assert "round=1" in s
        assert "Players (3)" in s


class TestGameStateIntegration:
    """Integration tests for GameState with all components."""

    @pytest.fixture
    def game_state(self) -> GameState:
        """Create a fully initialized game state for testing."""
        # Create board
        graph = BoardGraph()
        for i in range(5):
            graph.nodes[i] = NodeState(
                node_id=i,
                building_slots=[BuildingSlot(zone=Zone.A)],
                is_central_park=(i == 0),
                is_train_station=(i == 4),
            )
        for i in range(4):
            edge_id = make_edge_id(i, i + 1)
            graph.edges[edge_id] = EdgeState(edge_id=edge_id)
            graph.adjacency.setdefault(i, set()).add(i + 1)
            graph.adjacency.setdefault(i + 1, set()).add(i)

        state = GameState.create_initial_state(graph, num_players=3)
        state.set_phase(Phase.CHOOSING_ACTIONS)
        return state

    def test_full_round_simulation(self, game_state: GameState):
        """Simulate a full choosing actions round."""
        # Each player places 2 markers (minimum requirement)
        placements = [
            (0, ActionAreaType.LINE_EXPANSION),
            (1, ActionAreaType.BUSES),
            (2, ActionAreaType.VRROOMM),
            (0, ActionAreaType.PASSENGERS),
            (1, ActionAreaType.BUILDINGS),
            (2, ActionAreaType.LINE_EXPANSION),
        ]

        for player_id, area_type in placements:
            player = game_state.get_player(player_id)
            player.place_marker()
            game_state.action_board.place_marker(area_type, player_id)

        # All players pass
        for player in game_state.players:
            player.pass_turn()

        assert game_state.all_players_passed()
        assert game_state.action_board.count_total_markers() == 6

        # Transition to resolving
        game_state.set_phase(Phase.RESOLVING_ACTIONS)

        # Check resolution order
        order = game_state.action_board.get_resolution_order()
        assert order[0] == ActionAreaType.LINE_EXPANSION

        # Get markers to resolve in first area
        markers = game_state.action_board.get_markers_to_resolve(ActionAreaType.LINE_EXPANSION)
        assert len(markers) == 2  # Two markers in LINE_EXPANSION

    def test_passenger_and_board_integration(self, game_state: GameState):
        """Test passenger movement through player network."""
        # Create passenger at central park (node 0)
        park = game_state.board.get_central_parks()[0]
        p = game_state.passenger_manager.create_passenger(location=park.node_id)
        park.add_passenger(p.passenger_id)

        # Player 0 builds rail network
        game_state.board.edges[(0, 1)].add_rail(0)
        game_state.board.edges[(1, 2)].add_rail(0)
        game_state.players[0].place_rail()
        game_state.players[0].place_rail()

        # Verify reachability
        reachable = game_state.board.get_reachable_nodes(0, start_node=0)
        assert reachable == {0, 1, 2}

        # Move passenger along network
        park.remove_passenger(p.passenger_id)
        game_state.passenger_manager.move_passenger(p.passenger_id, new_location=2)
        game_state.board.get_node(2).add_passenger(p.passenger_id)

        # Validate state consistency
        errors = game_state.validate()
        assert len(errors) == 0

    def test_time_clock_integration(self, game_state: GameState):
        """Test time clock advancement with player time stones."""
        # Initial state
        assert game_state.global_state.time_clock_position == BuildingType.HOUSE

        # Started with 5 (default for 3-player game created in fixture is 4, wait let's check fixture)
        # The fixture uses create_initial_state(..., num_players=3) so it should have 4 stones.
        
        # Check initial value first to be robust
        initial_stones = game_state.global_state.time_stones_remaining
        
        # Player takes time stone (reverses clock)
        game_state.global_state.take_time_stone()
        game_state.players[0].take_time_stone()

        assert game_state.global_state.time_stones_remaining == initial_stones - 1
        assert game_state.players[0].time_stones == 1

        # Advance time clock normally
        game_state.global_state.advance_time_clock()
        assert game_state.global_state.time_clock_position == BuildingType.OFFICE

    def test_clone_preserves_complex_state(self, game_state: GameState):
        """Clone should preserve all component relationships."""
        # Set up complex state
        game_state.players[0].add_score(15)
        game_state.players[1].pass_turn()
        game_state.action_board.place_marker(ActionAreaType.VRROOMM, player_id=2)
        game_state.board.edges[(1, 2)].add_rail(0)
        p = game_state.passenger_manager.create_passenger(location=3)
        game_state.board.get_node(3).add_passenger(p.passenger_id)
        game_state.global_state.advance_time_clock()

        # Clone
        clone = game_state.clone()

        # Verify all state preserved
        assert clone.players[0].score == 15
        assert clone.players[1].has_passed
        assert clone.action_board.count_total_markers() == 1
        assert clone.board.edges[(1, 2)].has_player_rail(0)
        assert clone.passenger_manager.count() == 1
        assert clone.global_state.time_clock_position == BuildingType.OFFICE

        # Verify independence
        game_state.players[0].add_score(5)
        assert clone.players[0].score == 15


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
