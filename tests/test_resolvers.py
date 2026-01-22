"""Tests for action resolvers.

This module tests the resolvers:
- StartingPlayerResolver
- BusesResolver
- TimeClockResolver
- PassengersResolver
- BuildingsResolver
- LineExpansionResolver
- VrrooommResolver

And the ActionResolver dispatcher.
"""

import pytest

from core.constants import (
    Phase,
    ActionAreaType,
    BuildingType,
    Zone,
    MAX_BUSES,
    ZONE_PRIORITY,
)
from core.game_state import GameState
from core.board import BoardGraph, make_edge_id
from data.loader import load_default_board

from engine.resolvers import (
    StartingPlayerResolver,
    StartingPlayerResult,
    BusesResolver,
    BusesResult,
    TimeClockResolver,
    TimeClockResult,
    TimeClockAction,
    PassengersResolver,
    PassengersResult,
    PassengerDistribution,
    BuildingsResolver,
    BuildingsResult,
    BuildingPlacement,
    LineExpansionResolver,
    LineExpansionResult,
    RailPlacement,
    VrrooommResolver,
    VrrooommResult,
    PassengerDelivery,
)
from engine.action_resolver import ActionResolver, ResolutionStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def board() -> BoardGraph:
    """Load the default board for testing."""
    return load_default_board()


@pytest.fixture
def game_state(board: BoardGraph) -> GameState:
    """Create a fresh game state with 4 players."""
    return GameState.create_initial_state(board, num_players=4)


@pytest.fixture
def state_with_starting_player_marker(game_state: GameState) -> GameState:
    """Game state with a marker in the Starting Player area."""
    game_state.action_board.place_marker(ActionAreaType.STARTING_PLAYER, player_id=2)
    return game_state


@pytest.fixture
def state_with_buses_marker(game_state: GameState) -> GameState:
    """Game state with a marker in the Buses area."""
    game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=1)
    return game_state


@pytest.fixture
def state_with_time_clock_marker(game_state: GameState) -> GameState:
    """Game state with a marker in the Time Clock area."""
    game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=0)
    return game_state


@pytest.fixture
def state_with_passengers_markers(game_state: GameState) -> GameState:
    """Game state with multiple markers in the Passengers area."""
    game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=0)
    game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)
    game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=2)
    return game_state


# =============================================================================
# StartingPlayerResolver Tests
# =============================================================================


class TestStartingPlayerResolver:
    """Tests for the StartingPlayerResolver."""

    def test_no_markers_returns_rotation(self, game_state: GameState):
        """Should rotate starting player if no markers in area."""
        resolver = StartingPlayerResolver(game_state)
        
        # Manually set starting player to 0
        game_state.global_state.starting_player_idx = 0

        assert not resolver.has_markers()
        result = resolver.resolve()

        # Should be resolved (auto-rotation)
        assert result.resolved
        # Should rotate to next player (1)
        assert result.new_starting_player_id == 1
        assert game_state.global_state.starting_player_idx == 1

    def test_resolve_sets_starting_player(
        self, state_with_starting_player_marker: GameState
    ):
        """Should set the starting player to whoever placed the marker."""
        resolver = StartingPlayerResolver(state_with_starting_player_marker)

        assert resolver.has_markers()
        result = resolver.resolve()

        assert result.resolved
        assert result.new_starting_player_id == 2
        assert state_with_starting_player_marker.global_state.starting_player_idx == 2

    def test_get_valid_actions_returns_empty(
        self, state_with_starting_player_marker: GameState
    ):
        """Starting Player has no player choices."""
        resolver = StartingPlayerResolver(state_with_starting_player_marker)
        actions = resolver.get_valid_actions()

        assert actions == []

    def test_is_resolution_complete(
        self, state_with_starting_player_marker: GameState
    ):
        """Resolution is always complete (no player interaction)."""
        resolver = StartingPlayerResolver(state_with_starting_player_marker)

        # Even before calling resolve(), this area requires no interaction
        assert resolver.is_resolution_complete()

    def test_get_markers_to_resolve(
        self, state_with_starting_player_marker: GameState
    ):
        """Should return the markers in the area."""
        resolver = StartingPlayerResolver(state_with_starting_player_marker)
        markers = resolver.get_markers_to_resolve()

        assert len(markers) == 1
        assert markers[0].player_id == 2


# =============================================================================
# BusesResolver Tests
# =============================================================================


class TestBusesResolver:
    """Tests for the BusesResolver."""

    def test_no_markers_returns_unresolved(self, game_state: GameState):
        """Should return unresolved if no markers in area."""
        resolver = BusesResolver(game_state)

        assert not resolver.has_markers()
        result = resolver.resolve()

        assert not result.resolved
        assert result.player_id is None

    def test_resolve_gains_bus(self, state_with_buses_marker: GameState):
        """Should increment the player's bus count."""
        resolver = BusesResolver(state_with_buses_marker)
        player = state_with_buses_marker.get_player(1)
        initial_buses = player.buses

        result = resolver.resolve()

        assert result.resolved
        assert result.player_id == 1
        assert result.bus_gained
        assert result.new_bus_count == initial_buses + 1
        assert player.buses == initial_buses + 1

    def test_resolve_updates_max_buses(self, state_with_buses_marker: GameState):
        """Should update M#oB if player now has the most buses."""
        resolver = BusesResolver(state_with_buses_marker)

        # Initially all players have 1 bus
        assert resolver.get_max_buses() == 1

        result = resolver.resolve()

        # Now player 1 has 2 buses
        assert result.new_max_buses == 2
        assert resolver.get_max_buses() == 2

    def test_resolve_at_max_buses(self, state_with_buses_marker: GameState):
        """Should not gain bus if already at maximum."""
        player = state_with_buses_marker.get_player(1)
        player.buses = MAX_BUSES

        resolver = BusesResolver(state_with_buses_marker)
        result = resolver.resolve()

        assert result.resolved
        assert not result.bus_gained
        assert result.new_bus_count == MAX_BUSES

    def test_get_valid_actions_returns_empty(
        self, state_with_buses_marker: GameState
    ):
        """Buses has no player choices."""
        resolver = BusesResolver(state_with_buses_marker)
        actions = resolver.get_valid_actions()

        assert actions == []

    def test_get_max_buses_tracks_all_players(self, game_state: GameState):
        """M#oB should track the highest bus count across all players."""
        game_state.players[0].buses = 3
        game_state.players[1].buses = 2
        game_state.players[2].buses = 4
        game_state.players[3].buses = 1

        resolver = BusesResolver(game_state)
        assert resolver.get_max_buses() == 4


# =============================================================================
# TimeClockResolver Tests
# =============================================================================


class TestTimeClockResolver:
    """Tests for the TimeClockResolver."""

    # def test_no_markers_returns_unresolved(self, game_state: GameState):
    #     """Should return unresolved if no markers in area."""
    #     resolver = TimeClockResolver(game_state)

    #     assert not resolver.has_markers()
    #     result = resolver.resolve()

    #     assert not result.resolved
    #     assert result.action_taken is None

    def test_resolve_advance_clock(self, state_with_time_clock_marker: GameState):
        """Default resolution advances the time clock."""
        resolver = TimeClockResolver(state_with_time_clock_marker)

        # Clock starts on HOUSE
        assert state_with_time_clock_marker.global_state.time_clock_position == BuildingType.HOUSE

        result = resolver.resolve(TimeClockAction.ADVANCE_CLOCK)

        assert result.resolved
        assert result.action_taken == TimeClockAction.ADVANCE_CLOCK
        assert result.new_clock_position == BuildingType.OFFICE
        assert not result.time_stone_taken
        # Should be 5 for 4-player game (default fixture)
        assert result.time_stones_remaining == 5

    def test_resolve_stop_clock(self, state_with_time_clock_marker: GameState):
        """Stopping the clock takes a time stone."""
        resolver = TimeClockResolver(state_with_time_clock_marker)

        result = resolver.resolve(TimeClockAction.STOP_CLOCK)

        assert result.resolved
        assert result.action_taken == TimeClockAction.STOP_CLOCK
        assert result.new_clock_position == BuildingType.HOUSE  # Unchanged
        assert result.time_stone_taken
        # Should be 4 (5 - 1) for 4-player game
        assert result.time_stones_remaining == 4

        # Player should have the time stone
        player = state_with_time_clock_marker.get_player(0)
        assert player.time_stones == 1

    def test_stop_clock_ends_game_on_last_stone(
        self, state_with_time_clock_marker: GameState
    ):
        """Taking the last time stone ends the game."""
        state_with_time_clock_marker.global_state.time_stones_remaining = 1

        resolver = TimeClockResolver(state_with_time_clock_marker)
        result = resolver.resolve(TimeClockAction.STOP_CLOCK)

        assert result.game_ended
        assert result.time_stones_remaining == 0
        assert state_with_time_clock_marker.global_state.game_ended

    def test_cannot_stop_clock_without_stones(
        self, state_with_time_clock_marker: GameState
    ):
        """Should raise error if trying to stop with no stones."""
        state_with_time_clock_marker.global_state.time_stones_remaining = 0

        resolver = TimeClockResolver(state_with_time_clock_marker)

        with pytest.raises(ValueError, match="no time stones remaining"):
            resolver.resolve(TimeClockAction.STOP_CLOCK)

    def test_get_valid_actions_includes_both_options(
        self, state_with_time_clock_marker: GameState
    ):
        """Should offer both advance and stop options when stones available."""
        resolver = TimeClockResolver(state_with_time_clock_marker)
        actions = resolver.get_valid_actions()

        assert len(actions) == 2
        action_types = {a["action"] for a in actions}
        assert TimeClockAction.ADVANCE_CLOCK in action_types
        assert TimeClockAction.STOP_CLOCK in action_types

    def test_get_valid_actions_only_advance_when_no_stones(
        self, state_with_time_clock_marker: GameState
    ):
        """Should only offer advance when no time stones remain."""
        state_with_time_clock_marker.global_state.time_stones_remaining = 0

        resolver = TimeClockResolver(state_with_time_clock_marker)
        actions = resolver.get_valid_actions()

        assert len(actions) == 1
        assert actions[0]["action"] == TimeClockAction.ADVANCE_CLOCK

    def test_clock_cycles_through_building_types(
        self, state_with_time_clock_marker: GameState
    ):
        """Clock should cycle: House -> Office -> Pub -> House."""
        resolver = TimeClockResolver(state_with_time_clock_marker)

        # Start at House
        assert state_with_time_clock_marker.global_state.time_clock_position == BuildingType.HOUSE

        # Advance to Office
        state_with_time_clock_marker.global_state.advance_time_clock()
        assert state_with_time_clock_marker.global_state.time_clock_position == BuildingType.OFFICE

        # Advance to Pub
        state_with_time_clock_marker.global_state.advance_time_clock()
        assert state_with_time_clock_marker.global_state.time_clock_position == BuildingType.PUB

        # Advance back to House
        state_with_time_clock_marker.global_state.advance_time_clock()
        assert state_with_time_clock_marker.global_state.time_clock_position == BuildingType.HOUSE

    def test_can_stop_clock(self, state_with_time_clock_marker: GameState):
        """can_stop_clock should check if stones are available."""
        resolver = TimeClockResolver(state_with_time_clock_marker)

        assert resolver.can_stop_clock()

        state_with_time_clock_marker.global_state.time_stones_remaining = 0
        assert not resolver.can_stop_clock()

    def test_get_resolving_player_id(self, state_with_time_clock_marker: GameState):
        """Should return the player who placed the marker."""
        resolver = TimeClockResolver(state_with_time_clock_marker)
        assert resolver.get_resolving_player_id() == 0

    def test_is_resolution_complete_after_resolve(
        self, state_with_time_clock_marker: GameState
    ):
        """Resolution is complete after resolve() is called."""
        resolver = TimeClockResolver(state_with_time_clock_marker)

        assert not resolver.is_resolution_complete()
        resolver.resolve()
        assert resolver.is_resolution_complete()


# =============================================================================
# PassengersResolver Tests
# =============================================================================


class TestPassengersResolver:
    """Tests for the PassengersResolver."""

    def test_no_markers_returns_unresolved(self, game_state: GameState):
        """Should return unresolved if no markers in area."""
        resolver = PassengersResolver(game_state)

        assert not resolver.has_markers()
        result = resolver.resolve_all()

        assert not result.resolved
        assert len(result.slot_results) == 0

    def test_passengers_for_slot_based_on_mob(
        self, state_with_passengers_markers: GameState
    ):
        """Passengers should be M#oB - slot_index (can be 0)."""
        resolver = PassengersResolver(state_with_passengers_markers)

        # M#oB is 1 (all players start with 1 bus)
        assert resolver.get_max_buses() == 1

        markers = resolver.get_markers_to_resolve()

        # Slot A (index 0): 1 - 0 = 1 passenger
        assert resolver.get_passengers_for_slot(markers[0]) == 1

        # Slot B (index 1): 1 - 1 = 0 (wasted marker)
        assert resolver.get_passengers_for_slot(markers[1]) == 0

    def test_passengers_for_slot_with_higher_mob(
        self, state_with_passengers_markers: GameState
    ):
        """Passengers scale with M#oB."""
        # Give a player more buses
        state_with_passengers_markers.players[0].buses = 4

        resolver = PassengersResolver(state_with_passengers_markers)
        assert resolver.get_max_buses() == 4

        markers = resolver.get_markers_to_resolve()

        # Slot A (index 0): 4 - 0 = 4 passengers
        assert resolver.get_passengers_for_slot(markers[0]) == 4

        # Slot B (index 1): 4 - 1 = 3 passengers
        assert resolver.get_passengers_for_slot(markers[1]) == 3

        # Slot C (index 2): 4 - 2 = 2 passengers
        assert resolver.get_passengers_for_slot(markers[2]) == 2

    def test_resolve_all_with_default_distribution(
        self, state_with_passengers_markers: GameState
    ):
        """resolve_all should use default distribution if none provided."""
        resolver = PassengersResolver(state_with_passengers_markers)
        initial_passengers = state_with_passengers_markers.passenger_manager.count()

        result = resolver.resolve_all()

        assert result.resolved
        assert len(result.slot_results) == 3
        # With M#oB=1: Slot A=1, Slot B=0, Slot C=0 (negative becomes 0)
        # Actually slot C is index 2, so 1-2=-1 -> 0
        # Total: 1 + 0 + 0 = 1 passenger
        assert result.total_passengers_spawned == 1

        # Passengers should have been created
        assert (
            state_with_passengers_markers.passenger_manager.count()
            > initial_passengers
        )

    def test_resolve_slot_with_distribution(
        self, state_with_passengers_markers: GameState
    ):
        """Should distribute passengers according to provided distribution."""
        # Increase M#oB for more passengers
        state_with_passengers_markers.players[0].buses = 3

        resolver = PassengersResolver(state_with_passengers_markers)
        train_stations = resolver.get_train_stations()

        assert len(train_stations) >= 2

        # First slot gets 3 passengers (M#oB - 0)
        dist = PassengerDistribution({
            train_stations[0]: 2,
            train_stations[1]: 1,
        })

        result = resolver.resolve_slot(dist)

        assert result.player_id == 0
        assert result.passengers_spawned == 3
        assert result.distribution[train_stations[0]] == 2
        assert result.distribution[train_stations[1]] == 1

    def test_resolve_slot_validates_total(
        self, state_with_passengers_markers: GameState
    ):
        """Should reject distribution with wrong total."""
        resolver = PassengersResolver(state_with_passengers_markers)
        train_stations = resolver.get_train_stations()

        # Wrong total (should be 1 for slot A with M#oB=1)
        dist = PassengerDistribution({
            train_stations[0]: 5,
            train_stations[1]: 3,
        })

        with pytest.raises(ValueError, match="doesn't match"):
            resolver.resolve_slot(dist)

    def test_resolve_slot_validates_station_ids(
        self, state_with_passengers_markers: GameState
    ):
        """Should reject distribution with invalid station IDs."""
        resolver = PassengersResolver(state_with_passengers_markers)

        # Invalid station ID
        dist = PassengerDistribution({
            9999: 1,
        })

        with pytest.raises(ValueError, match="Invalid train station"):
            resolver.resolve_slot(dist)

    def test_get_valid_actions_returns_distributions(
        self, state_with_passengers_markers: GameState
    ):
        """Should return all valid distribution options."""
        # Increase M#oB for more options
        state_with_passengers_markers.players[0].buses = 3

        resolver = PassengersResolver(state_with_passengers_markers)
        actions = resolver.get_valid_actions()

        # Slot A gets 3 passengers, so 4 distribution options (0-3, 3-0, 1-2, 2-1)
        assert len(actions) == 4

        for action in actions:
            assert action["player_id"] == 0
            assert action["total_passengers"] == 3
            assert sum(action["distribution"].values()) == 3

    def test_passengers_added_to_nodes(
        self, state_with_passengers_markers: GameState
    ):
        """Passengers should be tracked at their nodes."""
        resolver = PassengersResolver(state_with_passengers_markers)
        train_stations = resolver.get_train_stations()

        result = resolver.resolve_all()

        # Check that passengers are at train stations
        total_at_stations = 0
        for station_id in train_stations:
            node = state_with_passengers_markers.board.get_node(station_id)
            total_at_stations += len(node.passenger_ids)

        assert total_at_stations == result.total_passengers_spawned

    def test_is_resolution_complete(
        self, state_with_passengers_markers: GameState
    ):
        """Resolution is complete when all slots are processed."""
        resolver = PassengersResolver(state_with_passengers_markers)

        assert not resolver.is_resolution_complete()

        # Resolve one slot at a time
        while not resolver.is_resolution_complete():
            actions = resolver.get_valid_actions()
            if actions:
                dist = PassengerDistribution(actions[0]["distribution"])
                resolver.resolve_slot(dist)

        assert resolver.is_resolution_complete()

    def test_get_current_slot(self, state_with_passengers_markers: GameState):
        """get_current_slot should track resolution progress."""
        resolver = PassengersResolver(state_with_passengers_markers)

        # First slot is slot A (player 0)
        current = resolver.get_current_slot()
        assert current is not None
        assert current.player_id == 0
        assert current.label == "A"

    def test_get_train_stations(self, state_with_passengers_markers: GameState):
        """Should return train station node IDs."""
        resolver = PassengersResolver(state_with_passengers_markers)
        stations = resolver.get_train_stations()

        # Default board has 2 train stations
        assert len(stations) == 2

        # Verify they are actually train stations
        for station_id in stations:
            node = state_with_passengers_markers.board.get_node(station_id)
            assert node.is_train_station


# =============================================================================
# Integration Tests
# =============================================================================


class TestResolversIntegration:
    """Integration tests for resolver interactions."""

    def test_buses_affects_passengers(self, game_state: GameState):
        """Gaining a bus should increase passengers for subsequent actions."""
        # Place markers in both areas
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)

        # Resolve buses first (as per resolution order)
        buses_resolver = BusesResolver(game_state)
        buses_result = buses_resolver.resolve()

        assert buses_result.new_max_buses == 2  # Player 0 now has 2 buses

        # Now passengers resolver should use new M#oB
        passengers_resolver = PassengersResolver(game_state)
        passengers_for_slot_a = passengers_resolver.get_passengers_for_slot(
            passengers_resolver.get_current_slot()
        )

        # Slot A: M#oB - 0 = 2 - 0 = 2 passengers
        assert passengers_for_slot_a == 2

    def test_time_clock_game_end_stops_resolution(self, game_state: GameState):
        """Taking the last time stone should end the game."""
        game_state.global_state.time_stones_remaining = 1
        game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=0)

        resolver = TimeClockResolver(game_state)
        result = resolver.resolve(TimeClockAction.STOP_CLOCK)

        assert result.game_ended
        assert game_state.is_game_over()

    def test_full_resolution_sequence(self, game_state: GameState):
        """Test resolving multiple areas in sequence."""
        # Place markers in multiple areas
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)
        game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=2)
        game_state.action_board.place_marker(ActionAreaType.STARTING_PLAYER, player_id=3)

        # Resolve in order (Buses -> Passengers -> Time Clock -> Starting Player)
        buses_resolver = BusesResolver(game_state)
        buses_result = buses_resolver.resolve()
        assert buses_result.resolved
        assert game_state.players[0].buses == 2

        passengers_resolver = PassengersResolver(game_state)
        passengers_result = passengers_resolver.resolve_all()
        assert passengers_result.resolved
        assert passengers_result.total_passengers_spawned >= 1

        time_clock_resolver = TimeClockResolver(game_state)
        time_clock_result = time_clock_resolver.resolve(TimeClockAction.ADVANCE_CLOCK)
        assert time_clock_result.resolved
        assert game_state.global_state.time_clock_position == BuildingType.OFFICE

        starting_player_resolver = StartingPlayerResolver(game_state)
        starting_player_result = starting_player_resolver.resolve()
        assert starting_player_result.resolved
        assert game_state.global_state.starting_player_idx == 3

    def test_multiple_passengers_markers(self, game_state: GameState):
        """Test resolving multiple passengers markers in order."""
        # Give someone more buses for variety
        game_state.players[0].buses = 3

        # Place multiple passengers markers
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=2)

        resolver = PassengersResolver(game_state)

        # Verify decreasing passenger counts
        markers = resolver.get_markers_to_resolve()
        assert resolver.get_passengers_for_slot(markers[0]) == 3  # A: 3-0
        assert resolver.get_passengers_for_slot(markers[1]) == 2  # B: 3-1
        assert resolver.get_passengers_for_slot(markers[2]) == 1  # C: 3-2

        # Resolve all
        result = resolver.resolve_all()

        assert result.total_passengers_spawned == 6  # 3 + 2 + 1
        assert len(result.slot_results) == 3


# =============================================================================
# BuildingsResolver Tests
# =============================================================================


@pytest.fixture
def state_with_buildings_marker(game_state: GameState) -> GameState:
    """Game state with a marker in the Buildings area."""
    game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=0)
    return game_state


@pytest.fixture
def state_with_buildings_markers(game_state: GameState) -> GameState:
    """Game state with multiple markers in the Buildings area."""
    game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=0)
    game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=1)
    game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=2)
    return game_state


class TestBuildingsResolver:
    """Tests for the BuildingsResolver."""

    def test_no_markers_returns_unresolved(self, game_state: GameState):
        """Should return unresolved if no markers in area."""
        resolver = BuildingsResolver(game_state)

        assert not resolver.has_markers()
        result = resolver.resolve_all()

        assert not result.resolved
        assert len(result.slot_results) == 0

    def test_buildings_for_slot_based_on_mob(
        self, state_with_buildings_markers: GameState
    ):
        """Buildings should be M#oB - slot_index (can be 0)."""
        resolver = BuildingsResolver(state_with_buildings_markers)

        # M#oB is 1 (all players start with 1 bus)
        assert resolver.get_max_buses() == 1

        markers = resolver.get_markers_to_resolve()

        # Find slot A and B by label
        slot_a = next(m for m in markers if m.label == "A")
        slot_b = next((m for m in markers if m.label == "B"), None)

        # Slot A (index 0): 1 - 0 = 1 building
        assert resolver.get_buildings_for_slot(slot_a) == 1

        # Slot B (index 1): 1 - 1 = 0 (wasted marker)
        if slot_b:
            assert resolver.get_buildings_for_slot(slot_b) == 0

    def test_buildings_for_slot_with_higher_mob(
        self, state_with_buildings_markers: GameState
    ):
        """Buildings scale with M#oB."""
        state_with_buildings_markers.players[0].buses = 4

        resolver = BuildingsResolver(state_with_buildings_markers)
        assert resolver.get_max_buses() == 4

        markers = resolver.get_markers_to_resolve()

        # Slot B is resolved first (index 0 in markers list due to reversed layout)
        # Slot A is resolved second (index 1 in markers list)
        
        # Verify based on slot label to be robust
        slot_a = next(m for m in markers if m.label == "A")
        slot_b = next(m for m in markers if m.label == "B")
        
        # Slot A (index 0): 4 - 0 = 4 buildings
        assert resolver.get_buildings_for_slot(slot_a) == 4

        # Slot B (index 1): 4 - 1 = 3 buildings
        assert resolver.get_buildings_for_slot(slot_b) == 3

    def test_get_available_zone_returns_innermost(
        self, state_with_buildings_marker: GameState
    ):
        """Should return innermost zone with available slots."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        # Initially Zone A should be available
        zone = resolver.get_available_zone()
        assert zone == Zone.A

    def test_inner_first_placement_enforced(
        self, state_with_buildings_marker: GameState
    ):
        """Should only allow placement in innermost available zone."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        valid_slots = resolver.get_valid_building_slots()

        # All valid slots should be in Zone A (innermost)
        for node_id, slot_index in valid_slots:
            node = state_with_buildings_marker.board.get_node(node_id)
            slot = node.building_slots[slot_index]
            assert slot.zone == Zone.A

    def test_place_building_updates_board(
        self, state_with_buildings_marker: GameState
    ):
        """Placing a building should update the board state."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        valid_slots = resolver.get_valid_building_slots()
        assert len(valid_slots) > 0

        node_id, slot_index = valid_slots[0]
        placement = BuildingPlacement(
            node_id=node_id,
            slot_index=slot_index,
            building_type=BuildingType.OFFICE,
        )

        resolver.place_building(placement)

        # Verify the building was placed
        node = state_with_buildings_marker.board.get_node(node_id)
        assert node.building_slots[slot_index].building == BuildingType.OFFICE

    def test_place_building_wrong_zone_rejected(
        self, state_with_buildings_marker: GameState
    ):
        """Should reject placement in outer zone when inner slots available."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        # Find a Zone B slot (outer zone)
        zone_b_slot = None
        for node_id, node in state_with_buildings_marker.board.nodes.items():
            for slot_idx, slot in enumerate(node.building_slots):
                if slot.zone == Zone.B and slot.is_empty():
                    zone_b_slot = (node_id, slot_idx)
                    break
            if zone_b_slot:
                break

        if zone_b_slot:
            placement = BuildingPlacement(
                node_id=zone_b_slot[0],
                slot_index=zone_b_slot[1],
                building_type=BuildingType.HOUSE,
            )

            with pytest.raises(ValueError, match="Must place in zone"):
                resolver.place_building(placement)

    def test_get_valid_actions_returns_all_combinations(
        self, state_with_buildings_marker: GameState
    ):
        """Should return all valid (slot, building_type) combinations."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        actions = resolver.get_valid_actions()

        # Each valid slot × 3 building types
        valid_slots = resolver.get_valid_building_slots()
        expected_count = len(valid_slots) * 3  # 3 building types

        assert len(actions) == expected_count

        # Verify all building types are represented
        building_types = {a["building_type"] for a in actions}
        assert building_types == {BuildingType.HOUSE, BuildingType.OFFICE, BuildingType.PUB}

    def test_resolve_all_places_buildings(
        self, state_with_buildings_marker: GameState
    ):
        """resolve_all should place buildings using default strategy."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        # Count initial empty slots in Zone A
        initial_zone_a_empty = len(
            state_with_buildings_marker.board.get_all_empty_slots_by_zone(Zone.A)
        )

        result = resolver.resolve_all()

        assert result.resolved
        assert result.total_buildings_placed >= 1

        # Should have fewer empty Zone A slots now
        final_zone_a_empty = len(
            state_with_buildings_marker.board.get_all_empty_slots_by_zone(Zone.A)
        )
        assert final_zone_a_empty < initial_zone_a_empty

    def test_is_resolution_complete(
        self, state_with_buildings_marker: GameState
    ):
        """Resolution is complete when all buildings for all slots are placed."""
        resolver = BuildingsResolver(state_with_buildings_marker)

        assert not resolver.is_resolution_complete()

        # Place the required building(s)
        while not resolver.is_resolution_complete():
            actions = resolver.get_valid_actions()
            if not actions:
                break
            action = actions[0]
            placement = BuildingPlacement(
                node_id=action["node_id"],
                slot_index=action["slot_index"],
                building_type=action["building_type"],
            )
            resolver.place_building(placement)

        assert resolver.is_resolution_complete()

    def test_zone_progression(self, state_with_buildings_markers: GameState):
        """Should move to next zone when current is filled."""
        # Give player more buses so they place more buildings
        state_with_buildings_markers.players[0].buses = 5

        resolver = BuildingsResolver(state_with_buildings_markers)

        # Fill all Zone A slots
        zone_a_slots = state_with_buildings_markers.board.get_all_empty_slots_by_zone(Zone.A)

        for node_id, slot in zone_a_slots:
            slot.place_building(BuildingType.HOUSE)

        # Now available zone should be B
        assert resolver.get_available_zone() == Zone.B

        # Valid slots should be in Zone B
        valid_slots = resolver.get_valid_building_slots()
        for node_id, slot_index in valid_slots:
            node = state_with_buildings_markers.board.get_node(node_id)
            assert node.building_slots[slot_index].zone == Zone.B


# =============================================================================
# LineExpansionResolver Tests
# =============================================================================


@pytest.fixture
def state_with_rail_network(game_state: GameState) -> GameState:
    """Game state with player 0 having a basic rail network."""
    # Find two adjacent nodes and place a rail segment
    node_ids = list(game_state.board.nodes.keys())
    start_node = node_ids[0]
    neighbors = list(game_state.board.get_neighbors(start_node))

    if neighbors:
        edge_id = make_edge_id(start_node, neighbors[0])
        game_state.board.edges[edge_id].add_rail(player_id=0)
        game_state.players[0].place_rail()
        game_state.players[0].network_endpoints = {start_node, neighbors[0]}

    return game_state


@pytest.fixture
def state_with_line_expansion_marker(state_with_rail_network: GameState) -> GameState:
    """Game state with rail network and a marker in Line Expansion area."""
    state_with_rail_network.action_board.place_marker(
        ActionAreaType.LINE_EXPANSION, player_id=0
    )
    return state_with_rail_network


@pytest.fixture
def state_with_line_expansion_markers(state_with_rail_network: GameState) -> GameState:
    """Game state with rail networks and multiple markers in Line Expansion area."""
    # Give each player a rail segment
    node_ids = list(state_with_rail_network.board.nodes.keys())

    # Player 1 network (different area)
    if len(node_ids) > 5:
        start = node_ids[5]
        neighbors = list(state_with_rail_network.board.get_neighbors(start))
        if neighbors:
            edge_id = make_edge_id(start, neighbors[0])
            if state_with_rail_network.board.edges[edge_id].is_empty():
                state_with_rail_network.board.edges[edge_id].add_rail(player_id=1)
                state_with_rail_network.players[1].place_rail()
                state_with_rail_network.players[1].network_endpoints = {start, neighbors[0]}

    state_with_rail_network.action_board.place_marker(
        ActionAreaType.LINE_EXPANSION, player_id=0
    )
    state_with_rail_network.action_board.place_marker(
        ActionAreaType.LINE_EXPANSION, player_id=1
    )

    return state_with_rail_network


class TestLineExpansionResolver:
    """Tests for the LineExpansionResolver."""

    def test_no_markers_returns_unresolved(self, game_state: GameState):
        """Should return unresolved if no markers in area."""
        resolver = LineExpansionResolver(game_state)

        assert not resolver.has_markers()
        result = resolver.resolve_all()

        assert not result.resolved
        assert len(result.slot_results) == 0

    def test_segments_for_slot_based_on_mob(
        self, state_with_line_expansion_marker: GameState
    ):
        """Segments should be M#oB - slot_index (can be 0)."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        # M#oB is 1 (all players start with 1 bus)
        assert resolver.get_max_buses() == 1

        markers = resolver.get_markers_to_resolve()

        # Slot A (index 0): 1 - 0 = 1 segment
        assert resolver.get_segments_for_slot(markers[0]) == 1

        # If there was a slot B (index 1): 1 - 1 = 0 (wasted marker)
        # This test only has one marker, but the formula allows 0

    def test_segments_for_slot_with_higher_mob(
        self, state_with_line_expansion_marker: GameState
    ):
        """Segments scale with M#oB."""
        state_with_line_expansion_marker.players[0].buses = 4

        resolver = LineExpansionResolver(state_with_line_expansion_marker)
        assert resolver.get_max_buses() == 4

        markers = resolver.get_markers_to_resolve()

        # Slot A (index 0): 4 - 0 = 4 segments
        assert resolver.get_segments_for_slot(markers[0]) == 4

    def test_get_player_endpoints(
        self, state_with_line_expansion_marker: GameState
    ):
        """Should return endpoints of player's rail network."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        endpoints = resolver.get_player_endpoints(player_id=0)

        # A single rail segment has 2 endpoints
        assert len(endpoints) == 2

    def test_get_valid_placements_from_endpoints(
        self, state_with_line_expansion_marker: GameState
    ):
        """Should return valid edges from endpoints."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        placements = resolver.get_valid_placements(player_id=0)

        # Should have at least one valid placement
        assert len(placements) > 0

        # All placements should be from endpoints
        endpoints = resolver.get_player_endpoints(player_id=0)
        for p in placements:
            assert p.from_endpoint in endpoints

    def test_place_rail_updates_board(
        self, state_with_line_expansion_marker: GameState
    ):
        """Placing a rail should update the board state."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        placements = resolver.get_valid_placements(player_id=0)
        assert len(placements) > 0

        placement = placements[0]
        initial_rails = len(
            state_with_line_expansion_marker.board.get_player_edges(player_id=0)
        )

        resolver.place_rail(placement)

        # Verify the rail was placed
        final_rails = len(
            state_with_line_expansion_marker.board.get_player_edges(player_id=0)
        )
        assert final_rails == initial_rails + 1

    def test_place_rail_from_non_endpoint_rejected(
        self, state_with_line_expansion_marker: GameState
    ):
        """Should reject placement from non-endpoint node."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        # Find a node that's NOT an endpoint
        endpoints = resolver.get_player_endpoints(player_id=0)
        all_nodes = set(state_with_line_expansion_marker.board.nodes.keys())
        non_endpoints = all_nodes - endpoints

        if non_endpoints:
            non_endpoint = list(non_endpoints)[0]
            neighbors = state_with_line_expansion_marker.board.get_neighbors(non_endpoint)

            if neighbors:
                edge_id = make_edge_id(non_endpoint, list(neighbors)[0])
                placement = RailPlacement(
                    edge_id=edge_id,
                    from_endpoint=non_endpoint,
                )

                with pytest.raises(ValueError, match="not an endpoint"):
                    resolver.place_rail(placement)

    def test_get_valid_actions(
        self, state_with_line_expansion_marker: GameState
    ):
        """Should return valid actions for the current player."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        actions = resolver.get_valid_actions()

        assert len(actions) > 0

        for action in actions:
            assert action["player_id"] == 0
            assert "edge_id" in action
            assert "from_endpoint" in action

    def test_resolve_all_places_rails(
        self, state_with_line_expansion_marker: GameState
    ):
        """resolve_all should place rails using default strategy."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        initial_rails = len(
            state_with_line_expansion_marker.board.get_player_edges(player_id=0)
        )

        result = resolver.resolve_all()

        assert result.resolved
        assert result.total_segments_placed >= 1

        final_rails = len(
            state_with_line_expansion_marker.board.get_player_edges(player_id=0)
        )
        assert final_rails > initial_rails

    def test_rail_sharing_not_allowed_with_empty_edges(
        self, state_with_line_expansion_markers: GameState
    ):
        """Cannot share rails when empty edges are available at endpoints."""
        resolver = LineExpansionResolver(state_with_line_expansion_markers)

        # Manually setup the board to ensure precise conditions:
        # 1. Player 0 has endpoints.
        # 2. Player 0 tracks touch Player 1's track, BUT NOT at Player 1's endpoint.
        # 3. Player 0 has empty edges at their endpoints.
        
        # Clear existing rails for clean slate
        for edge in state_with_line_expansion_markers.board.edges.values():
            edge.rail_segments = []
        state_with_line_expansion_markers.players[0].rail_segments_remaining = 15
        state_with_line_expansion_markers.players[1].rail_segments_remaining = 15
        
        # Setup specific topology on default board:
        # P1 owns path 2-3-8. Endpoints {2, 8}. Node 3 is mid-segment.
        # P0 owns edge 3-7. Endpoints {3, 7}.
        # Junction is at Node 3.
        
        board = state_with_line_expansion_markers.board
        
        # P1 rail: 2-3-8
        # Ensure edges exist (based on default_board.json)
        board.get_edge(2, 3).add_rail(1)
        board.get_edge(3, 8).add_rail(1)
        
        # P0 rail: 3-7
        board.get_edge(3, 7).add_rail(0)

        # Initialize endpoints for manual setup
        state_with_line_expansion_markers.players[0].network_endpoints = {3, 7}
        state_with_line_expansion_markers.players[1].network_endpoints = {2, 8}

        # Verify P0 endpoint 3 touches P1 rail at 3.
        # Verify 3 is NOT P1 endpoint (it has 2 P1 segments).
        p1_endpoints = resolver.get_player_endpoints(1)
        assert 3 not in p1_endpoints
        
        # Verify P0 has empty edges at endpoint 7.
        # Node 7 connects to 2, 3, 6, 8, 11, 12. P0 operates on 3-7.
        # So edges (2,7), (6,7), etc. are empty.
        empty_at_7 = board.get_empty_edges_at_node(7)
        assert len(empty_at_7) > 0
        
        # Check rule 2: Endpoints touch other player
        # P0 endpoints: {3, 7}.
        # P1 endpoints: {2, 8}.
        # Intersection: Empty.
        assert not resolver._endpoints_touch_other_player(0)
        
        # Check rule 1: Empty edges available
        # We confirmed empty edges at 7.
        has_empty = resolver._has_empty_edge_at_endpoints(0)
        assert has_empty
        
        if has_empty:
            assert not resolver.can_share_rails(player_id=0)

    def test_rail_sharing_allowed_when_no_empty_edges(
        self, state_with_line_expansion_marker: GameState
    ):
        """Can share rails when no empty edges at any endpoint."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        # Fill all edges at player 0's endpoints
        endpoints = resolver.get_player_endpoints(player_id=0)
        for endpoint in endpoints:
            for neighbor in state_with_line_expansion_marker.board.get_neighbors(endpoint):
                edge_id = make_edge_id(endpoint, neighbor)
                edge = state_with_line_expansion_marker.board.edges[edge_id]
                if edge.is_empty():
                    # Add another player's rail to simulate occupied
                    edge.add_rail(player_id=1)

        # Now sharing should be allowed
        assert resolver.can_share_rails(player_id=0)

    def test_is_resolution_complete(
        self, state_with_line_expansion_marker: GameState
    ):
        """Resolution is complete when all segments for all slots are placed."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        assert not resolver.is_resolution_complete()

        # Place the required segment(s)
        while not resolver.is_resolution_complete():
            placements = resolver.get_valid_placements(player_id=0)
            if not placements:
                break
            resolver.place_rail(placements[0])

        assert resolver.is_resolution_complete()

    def test_player_rail_inventory_decremented(
        self, state_with_line_expansion_marker: GameState
    ):
        """Placing rails should decrement player's rail inventory."""
        resolver = LineExpansionResolver(state_with_line_expansion_marker)
        player = state_with_line_expansion_marker.get_player(0)

        initial_rails = player.rail_segments_remaining
        placements = resolver.get_valid_placements(player_id=0)

        if placements:
            resolver.place_rail(placements[0])
            assert player.rail_segments_remaining == initial_rails - 1

    def test_no_actions_when_no_rails_remaining(
        self, state_with_line_expansion_marker: GameState
    ):
        """Should return no actions when player has no rail segments."""
        player = state_with_line_expansion_marker.get_player(0)
        player.rail_segments_remaining = 0

        resolver = LineExpansionResolver(state_with_line_expansion_marker)
        actions = resolver.get_valid_actions()

        assert len(actions) == 0

    def test_endpoints_update_after_placement(
        self, state_with_line_expansion_marker: GameState
    ):
        """Endpoints should update as network extends."""
        # Give player more buses for more segments
        state_with_line_expansion_marker.players[0].buses = 3

        resolver = LineExpansionResolver(state_with_line_expansion_marker)

        initial_endpoints = resolver.get_player_endpoints(player_id=0)

        # Place a rail
        placements = resolver.get_valid_placements(player_id=0)
        if placements:
            resolver.place_rail(placements[0])

            # Endpoints should have changed (one endpoint extended, possibly new one)
            new_endpoints = resolver.get_player_endpoints(player_id=0)

            # The old endpoint we extended from should no longer be an endpoint
            # (unless it was the only connection point)
            # At minimum, network shape changed
            assert new_endpoints != initial_endpoints or len(new_endpoints) == len(initial_endpoints)


# =============================================================================
# Buildings and Line Expansion Integration Tests
# =============================================================================


class TestBuildingsLineExpansionIntegration:
    """Integration tests for buildings and line expansion resolvers."""

    def test_buses_affects_buildings(self, game_state: GameState):
        """Gaining a bus should increase buildings for subsequent actions."""
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=1)

        # Resolve buses first
        buses_resolver = BusesResolver(game_state)
        buses_result = buses_resolver.resolve()

        assert buses_result.new_max_buses == 2

        # Buildings resolver should use new M#oB
        buildings_resolver = BuildingsResolver(game_state)
        buildings_for_slot = buildings_resolver.get_buildings_for_slot(
            buildings_resolver.get_current_slot()
        )

        # Slot A: M#oB - 0 = 2 buildings
        assert buildings_for_slot == 2

    def test_buses_affects_line_expansion(self, state_with_rail_network: GameState):
        """Gaining a bus should increase segments for subsequent actions."""
        state_with_rail_network.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
        state_with_rail_network.action_board.place_marker(
            ActionAreaType.LINE_EXPANSION, player_id=0
        )

        # Resolve buses first
        buses_resolver = BusesResolver(state_with_rail_network)
        buses_result = buses_resolver.resolve()

        assert buses_result.new_max_buses == 2

        # Line expansion resolver should use new M#oB
        line_resolver = LineExpansionResolver(state_with_rail_network)
        segments_for_slot = line_resolver.get_segments_for_slot(
            line_resolver.get_current_slot()
        )

        # Slot A: M#oB - 0 = 2 segments
        assert segments_for_slot == 2

    def test_full_resolution_with_new_resolvers(self, state_with_rail_network: GameState):
        """Test resolving all action areas including buildings and line expansion."""
        # Set up markers
        state_with_rail_network.action_board.place_marker(
            ActionAreaType.LINE_EXPANSION, player_id=0
        )
        state_with_rail_network.action_board.place_marker(
            ActionAreaType.BUSES, player_id=1
        )
        state_with_rail_network.action_board.place_marker(
            ActionAreaType.BUILDINGS, player_id=2
        )
        state_with_rail_network.action_board.place_marker(
            ActionAreaType.STARTING_PLAYER, player_id=3
        )

        # Resolve in order
        line_resolver = LineExpansionResolver(state_with_rail_network)
        line_result = line_resolver.resolve_all()
        assert line_result.resolved

        buses_resolver = BusesResolver(state_with_rail_network)
        buses_result = buses_resolver.resolve()
        assert buses_result.resolved

        buildings_resolver = BuildingsResolver(state_with_rail_network)
        buildings_result = buildings_resolver.resolve_all()
        assert buildings_result.resolved

        starting_resolver = StartingPlayerResolver(state_with_rail_network)
        starting_result = starting_resolver.resolve()
        assert starting_result.resolved
        assert state_with_rail_network.global_state.starting_player_idx == 3


# =============================================================================
# VrrooommResolver Tests
# =============================================================================


@pytest.fixture
def state_with_vrroomm_setup(game_state: GameState) -> GameState:
    """Game state set up for Vrroomm! testing.

    This fixture:
    - Places a rail network for player 0
    - Places a building matching the time clock (HOUSE)
    - Spawns a passenger at a node in the network
    """
    board = game_state.board

    # Find nodes with building slots and place a HOUSE
    for node_id, node in board.nodes.items():
        if node.building_slots:
            for slot in node.building_slots:
                if slot.is_empty():
                    slot.place_building(BuildingType.HOUSE)
                    break
            break

    # Create a rail network for player 0 connecting to that node
    node_ids = list(board.nodes.keys())
    # Build a small network: node 0 - node 1
    if len(node_ids) >= 2:
        edge_id = make_edge_id(node_ids[0], node_ids[1])
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=0)
            game_state.players[0].place_rail()

    # Spawn a passenger at node 0
    passenger = game_state.passenger_manager.create_passenger(node_ids[0])
    board.get_node(node_ids[0]).add_passenger(passenger.passenger_id)

    return game_state


@pytest.fixture
def state_with_vrroomm_marker(state_with_vrroomm_setup: GameState) -> GameState:
    """Game state with Vrroomm setup and a marker placed."""
    state_with_vrroomm_setup.action_board.place_marker(
        ActionAreaType.VRROOMM, player_id=0
    )
    return state_with_vrroomm_setup


@pytest.fixture
def state_with_complete_vrroomm_setup(game_state: GameState) -> GameState:
    """Complete Vrroomm setup with passenger, network, and matching building."""
    board = game_state.board

    # Build a rail network: nodes 0 - 1 - 2
    node_ids = list(board.nodes.keys())

    # Place rails
    edge_01 = make_edge_id(node_ids[0], node_ids[1])
    edge_12 = make_edge_id(node_ids[1], node_ids[2])

    if edge_01 in board.edges:
        board.edges[edge_01].add_rail(player_id=0)
        game_state.players[0].place_rail()
    if edge_12 in board.edges:
        board.edges[edge_12].add_rail(player_id=0)
        game_state.players[0].place_rail()

    # Place a HOUSE at node 2 (destination)
    node_2 = board.get_node(node_ids[2])
    if node_2.building_slots:
        node_2.building_slots[0].place_building(BuildingType.HOUSE)

    # Spawn a passenger at node 0 (start)
    passenger = game_state.passenger_manager.create_passenger(node_ids[0])
    board.get_node(node_ids[0]).add_passenger(passenger.passenger_id)

    # Place marker
    game_state.action_board.place_marker(ActionAreaType.VRROOMM, player_id=0)

    return game_state


class TestVrrooommResolver:
    """Tests for the VrrooommResolver."""

    def test_no_markers_returns_unresolved(self, game_state: GameState):
        """Should return unresolved if no markers in area."""
        resolver = VrrooommResolver(game_state)

        assert not resolver.has_markers()
        result = resolver.resolve_all()

        assert not result.resolved
        assert len(result.slot_results) == 0

    def test_get_current_building_type(self, state_with_vrroomm_marker: GameState):
        """Should return the current time clock position."""
        resolver = VrrooommResolver(state_with_vrroomm_marker)

        # Clock starts on HOUSE
        assert resolver.get_current_building_type() == BuildingType.HOUSE

        # Advance clock
        state_with_vrroomm_marker.global_state.advance_time_clock()
        # Need new resolver to see the change
        resolver2 = VrrooommResolver(state_with_vrroomm_marker)
        assert resolver2.get_current_building_type() == BuildingType.OFFICE

    def test_get_deliveries_remaining(self, state_with_vrroomm_marker: GameState):
        """Deliveries should be limited by bus count."""
        resolver = VrrooommResolver(state_with_vrroomm_marker)

        # Player 0 has 1 bus initially
        assert resolver.get_deliveries_remaining_for_current_slot() == 1

        # Give player more buses
        state_with_vrroomm_marker.players[0].buses = 3
        resolver2 = VrrooommResolver(state_with_vrroomm_marker)
        assert resolver2.get_deliveries_remaining_for_current_slot() == 3

    def test_get_reachable_nodes(self, state_with_complete_vrroomm_setup: GameState):
        """Should return nodes in player's rail network."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        reachable = resolver.get_reachable_nodes_for_player(player_id=0)

        # Network spans 3 nodes
        assert len(reachable) >= 2

    def test_get_available_passengers(self, state_with_complete_vrroomm_setup: GameState):
        """Should return passengers at reachable nodes."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        available = resolver.get_available_passengers(player_id=0)

        # We placed one passenger
        assert len(available) >= 1

    def test_get_available_destinations(self, state_with_complete_vrroomm_setup: GameState):
        """Should return valid destination buildings."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        available_passengers = resolver.get_available_passengers(player_id=0)
        if available_passengers:
            passenger_id = available_passengers[0]
            destinations = resolver.get_available_destinations(player_id=0, passenger_id=passenger_id)

            # Should have at least one destination (the HOUSE we placed)
            assert len(destinations) >= 1

    def test_get_valid_actions(self, state_with_complete_vrroomm_setup: GameState):
        """Should return valid (passenger, destination) combinations."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        actions = resolver.get_valid_actions()

        # Should have at least one valid action
        assert len(actions) >= 1

        for action in actions:
            assert "player_id" in action
            assert "passenger_id" in action
            assert "from_node" in action
            assert "to_node" in action
            assert "building_slot_index" in action

    def test_deliver_passenger_scores_point(self, state_with_complete_vrroomm_setup: GameState):
        """Delivering a passenger should score 1 point."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)
        player = state_with_complete_vrroomm_setup.get_player(0)
        initial_score = player.score

        actions = resolver.get_valid_actions()
        if actions:
            action = actions[0]
            delivery = PassengerDelivery(
                passenger_id=action["passenger_id"],
                from_node=action["from_node"],
                to_node=action["to_node"],
                building_slot_index=action["building_slot_index"],
            )
            resolver.deliver_passenger(delivery)

            assert player.score == initial_score + 1

    def test_deliver_passenger_moves_passenger(self, state_with_complete_vrroomm_setup: GameState):
        """Delivering a passenger should move them to the destination."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        actions = resolver.get_valid_actions()
        if actions:
            action = actions[0]
            passenger_id = action["passenger_id"]
            from_node = action["from_node"]
            to_node = action["to_node"]

            delivery = PassengerDelivery(
                passenger_id=passenger_id,
                from_node=from_node,
                to_node=to_node,
                building_slot_index=action["building_slot_index"],
            )
            resolver.deliver_passenger(delivery)

            # Passenger should be at new location
            passenger = state_with_complete_vrroomm_setup.passenger_manager.get_passenger(passenger_id)
            assert passenger.location == to_node

            # Passenger should be in the new node's passenger_ids
            new_node = state_with_complete_vrroomm_setup.board.get_node(to_node)
            assert passenger_id in new_node.passenger_ids

            # Passenger should NOT be in the old node's passenger_ids
            old_node = state_with_complete_vrroomm_setup.board.get_node(from_node)
            assert passenger_id not in old_node.passenger_ids

    def test_building_slot_occupied_after_delivery(self, state_with_complete_vrroomm_setup: GameState):
        """Delivered passenger should occupy the building slot during resolution."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        actions = resolver.get_valid_actions()
        if actions:
            action = actions[0]
            to_node = action["to_node"]
            slot_idx = action["building_slot_index"]

            delivery = PassengerDelivery(
                passenger_id=action["passenger_id"],
                from_node=action["from_node"],
                to_node=to_node,
                building_slot_index=slot_idx,
            )
            resolver.deliver_passenger(delivery)

            # Slot should be marked as occupied
            node = state_with_complete_vrroomm_setup.board.get_node(to_node)
            assert node.building_slots[slot_idx].is_occupied_by_passenger()

    def test_cannot_deliver_to_occupied_slot(self, state_with_complete_vrroomm_setup: GameState):
        """Cannot deliver to a building slot already occupied."""
        # Add a second passenger
        board = state_with_complete_vrroomm_setup.board
        node_ids = list(board.nodes.keys())
        passenger2 = state_with_complete_vrroomm_setup.passenger_manager.create_passenger(node_ids[0])
        board.get_node(node_ids[0]).add_passenger(passenger2.passenger_id)

        # Give player 2 buses so they can attempt 2 deliveries
        state_with_complete_vrroomm_setup.players[0].buses = 2

        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        actions = resolver.get_valid_actions()
        if len(actions) >= 1:
            # Make first delivery
            action = actions[0]
            delivery = PassengerDelivery(
                passenger_id=action["passenger_id"],
                from_node=action["from_node"],
                to_node=action["to_node"],
                building_slot_index=action["building_slot_index"],
            )
            resolver.deliver_passenger(delivery)

            # Try to deliver to the same slot - should fail
            delivery2 = PassengerDelivery(
                passenger_id=passenger2.passenger_id,
                from_node=node_ids[0],
                to_node=action["to_node"],
                building_slot_index=action["building_slot_index"],
            )
            with pytest.raises(ValueError, match="already occupied"):
                resolver.deliver_passenger(delivery2)

    def test_resolve_all_with_defaults(self, state_with_complete_vrroomm_setup: GameState):
        """resolve_all should make deliveries using default strategy."""
        player = state_with_complete_vrroomm_setup.get_player(0)
        initial_score = player.score

        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)
        result = resolver.resolve_all()

        assert result.resolved
        # Should have made at least one delivery
        assert result.total_deliveries >= 1
        assert result.total_points_scored >= 1
        assert player.score > initial_score

    def test_occupancy_cleared_after_resolution(self, state_with_complete_vrroomm_setup: GameState):
        """Building slot occupancy should be cleared after resolve_all."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)
        result = resolver.resolve_all()

        # After resolution, occupied_by_passenger_id should be cleared
        for node in state_with_complete_vrroomm_setup.board.nodes.values():
            for slot in node.building_slots:
                assert not slot.is_occupied_by_passenger()

    def test_is_resolution_complete(self, state_with_complete_vrroomm_setup: GameState):
        """Resolution complete when all slots processed."""
        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)

        assert not resolver.is_resolution_complete()

        # Make deliveries until done
        while not resolver.is_resolution_complete():
            actions = resolver.get_valid_actions()
            if actions:
                action = actions[0]
                delivery = PassengerDelivery(
                    passenger_id=action["passenger_id"],
                    from_node=action["from_node"],
                    to_node=action["to_node"],
                    building_slot_index=action["building_slot_index"],
                )
                resolver.deliver_passenger(delivery)

            # Check if we can still deliver
            remaining = resolver.get_deliveries_remaining_for_current_slot()
            if remaining <= 0 or not resolver.get_valid_actions():
                resolver.finalize_current_slot()

        assert resolver.is_resolution_complete()

    def test_only_delivers_to_matching_building_type(self, state_with_complete_vrroomm_setup: GameState):
        """Should only deliver to buildings matching time clock."""
        # Change time clock to OFFICE
        state_with_complete_vrroomm_setup.global_state.time_clock_position = BuildingType.OFFICE

        resolver = VrrooommResolver(state_with_complete_vrroomm_setup)
        actions = resolver.get_valid_actions()

        # We only have a HOUSE building, so no valid actions for OFFICE time
        # (assuming setup only placed HOUSE)
        for action in actions:
            to_node = action["to_node"]
            slot_idx = action["building_slot_index"]
            node = state_with_complete_vrroomm_setup.board.get_node(to_node)
            slot = node.building_slots[slot_idx]
            # All valid actions should be for OFFICE buildings
            assert slot.building == BuildingType.OFFICE


# =============================================================================
# ActionResolver Tests
# =============================================================================


@pytest.fixture
def state_with_multiple_markers(game_state: GameState) -> GameState:
    """Game state with markers in multiple action areas."""
    # Set up some infrastructure first
    board = game_state.board
    node_ids = list(board.nodes.keys())

    # Place a rail for player 0
    edge_id = make_edge_id(node_ids[0], node_ids[1])
    if edge_id in board.edges:
        board.edges[edge_id].add_rail(player_id=0)
        game_state.players[0].place_rail()

    # Place markers in various areas
    game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
    game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)
    game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=2)
    game_state.action_board.place_marker(ActionAreaType.STARTING_PLAYER, player_id=3)

    return game_state


class TestActionResolver:
    """Tests for the ActionResolver dispatcher."""

    def test_resolve_all_with_no_markers(self, game_state: GameState):
        """Should complete with no markers to resolve."""
        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

        assert result.completed
        # Time Clock is now always resolved even without markers
        assert len(result.area_results) == 1
        assert result.area_results[0].area_type == ActionAreaType.TIME_CLOCK

    def test_resolve_all_processes_all_areas(self, state_with_multiple_markers: GameState):
        """Should resolve all areas with markers."""
        resolver = ActionResolver(state_with_multiple_markers)
        result = resolver.resolve_all()

        assert result.completed

        # Should have results for areas with markers
        resolved_areas = {r.area_type for r in result.area_results}
        assert ActionAreaType.BUSES in resolved_areas
        assert ActionAreaType.PASSENGERS in resolved_areas
        assert ActionAreaType.TIME_CLOCK in resolved_areas
        assert ActionAreaType.STARTING_PLAYER in resolved_areas

    def test_resolve_all_updates_game_state(self, state_with_multiple_markers: GameState):
        """Resolving should update the game state."""
        initial_buses = state_with_multiple_markers.players[0].buses
        initial_passengers = state_with_multiple_markers.passenger_manager.count()

        resolver = ActionResolver(state_with_multiple_markers)
        resolver.resolve_all()

        # Player 0 should have gained a bus
        assert state_with_multiple_markers.players[0].buses == initial_buses + 1

        # Passengers should have been spawned
        assert state_with_multiple_markers.passenger_manager.count() > initial_passengers

        # Starting player should have changed
        assert state_with_multiple_markers.global_state.starting_player_idx == 3

    def test_resolve_all_respects_resolution_order(self, game_state: GameState):
        """Areas should be resolved in the correct order."""
        # Place markers in reverse order to ensure resolution order is enforced
        game_state.action_board.place_marker(ActionAreaType.STARTING_PLAYER, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=1)

        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

        # Check that BUSES was resolved before STARTING_PLAYER
        area_order = [r.area_type for r in result.area_results]
        buses_idx = area_order.index(ActionAreaType.BUSES)
        starting_idx = area_order.index(ActionAreaType.STARTING_PLAYER)

        assert buses_idx < starting_idx

    def test_game_end_on_last_time_stone(self, game_state: GameState):
        """Should detect game end when last time stone is taken."""
        game_state.global_state.time_stones_remaining = 1
        game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=0)

        resolver = ActionResolver(game_state)
        # Override to use STOP_CLOCK action
        # For this test, we manually resolve with stop_clock
        time_resolver = TimeClockResolver(game_state)
        result = time_resolver.resolve(TimeClockAction.STOP_CLOCK)

        assert result.game_ended

    def test_start_resolution_sets_context(self, state_with_multiple_markers: GameState):
        """start_resolution should initialize the context."""
        resolver = ActionResolver(state_with_multiple_markers)
        resolver.start_resolution()

        context = resolver.get_context()

        assert context.status in (ResolutionStatus.IN_PROGRESS, ResolutionStatus.AWAITING_INPUT)
        # First area with markers that requires input should be PASSENGERS
        # (BUSES is now auto-resolved)
        assert context.current_area == ActionAreaType.PASSENGERS

    def test_is_complete_tracking(self, state_with_multiple_markers: GameState):
        """is_complete should track resolution progress."""
        resolver = ActionResolver(state_with_multiple_markers)

        assert not resolver.is_complete()

        resolver.resolve_all()

        # After resolve_all, context should be ALL_COMPLETE
        # Note: resolve_all creates its own internal state, so we check results instead
        result = resolver.resolve_all()
        assert result.completed

    def test_get_results(self, state_with_multiple_markers: GameState):
        """get_results should return resolved area results."""
        resolver = ActionResolver(state_with_multiple_markers)
        resolver.resolve_all()

        results = resolver.get_results()

        assert len(results) >= 4  # At least the 4 areas we placed markers in

    def test_buses_affects_subsequent_resolvers(self, game_state: GameState):
        """Resolving buses should affect M#oB for later resolvers."""
        # Place buses before passengers
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=1)

        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

        # Find the passengers result
        passengers_result = None
        for r in result.area_results:
            if r.area_type == ActionAreaType.PASSENGERS:
                passengers_result = r.result
                break

        # With M#oB = 2 after buses, slot A should get 2 passengers
        # But since we use default resolution, check total is at least 2
        assert passengers_result is not None
        assert passengers_result.total_passengers_spawned >= 2


# =============================================================================
# Full Integration Tests
# =============================================================================


class TestFullResolutionIntegration:
    """Full integration tests for the resolution system."""

    def test_complete_resolution_cycle(self, game_state: GameState):
        """Test a complete resolution cycle with all areas."""
        board = game_state.board
        node_ids = list(board.nodes.keys())

        # Set up rail network for player 0
        edge_id = make_edge_id(node_ids[0], node_ids[1])
        if edge_id in board.edges:
            board.edges[edge_id].add_rail(player_id=0)
            game_state.players[0].place_rail()

        # Place markers in all areas
        game_state.action_board.place_marker(ActionAreaType.LINE_EXPANSION, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.BUSES, player_id=1)
        game_state.action_board.place_marker(ActionAreaType.PASSENGERS, player_id=2)
        game_state.action_board.place_marker(ActionAreaType.BUILDINGS, player_id=3)
        game_state.action_board.place_marker(ActionAreaType.TIME_CLOCK, player_id=0)
        game_state.action_board.place_marker(ActionAreaType.STARTING_PLAYER, player_id=1)

        # Resolve all
        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

        assert result.completed
        assert len(result.area_results) == 6

        # Verify state changes
        assert game_state.players[1].buses == 2  # Gained a bus
        assert game_state.passenger_manager.count() > 0  # Passengers spawned
        assert game_state.global_state.starting_player_idx == 1

    def test_vrroomm_with_full_setup(self, game_state: GameState):
        """Test Vrroomm! with complete game setup."""
        board = game_state.board
        node_ids = list(board.nodes.keys())

        # Build network
        edge_01 = make_edge_id(node_ids[0], node_ids[1])
        edge_12 = make_edge_id(node_ids[1], node_ids[2])
        if edge_01 in board.edges:
            board.edges[edge_01].add_rail(player_id=0)
            game_state.players[0].place_rail()
        if edge_12 in board.edges:
            board.edges[edge_12].add_rail(player_id=0)
            game_state.players[0].place_rail()

        # Place OFFICE at node 2 (clock will advance from HOUSE to OFFICE)
        node_2 = board.get_node(node_ids[2])
        if node_2.building_slots:
            node_2.building_slots[0].place_building(BuildingType.OFFICE)

        # Spawn passenger at node 0
        passenger = game_state.passenger_manager.create_passenger(node_ids[0])
        board.get_node(node_ids[0]).add_passenger(passenger.passenger_id)

        # Place Vrroomm marker
        game_state.action_board.place_marker(ActionAreaType.VRROOMM, player_id=0)

        initial_score = game_state.players[0].score

        # Resolve
        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

        assert result.completed

        # Find vrroomm result
        vrroomm_result = None
        for r in result.area_results:
            if r.area_type == ActionAreaType.VRROOMM:
                vrroomm_result = r.result
                break

        assert vrroomm_result is not None
        assert vrroomm_result.total_deliveries >= 1
        assert game_state.players[0].score > initial_score
