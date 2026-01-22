"""Action resolver dispatcher for the Bus game engine.

The ActionResolver is responsible for coordinating the resolution of all
action areas during the Resolving Actions phase. It dispatches to individual
resolvers in the correct order and manages state transitions between areas.

Resolution order (fixed):
1. Line Expansion
2. Buses
3. Passengers
4. Buildings
5. Time Clock
6. Vrroomm!
7. Starting Player
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from enum import Enum

from core.constants import ActionAreaType, ACTION_RESOLUTION_ORDER

from .resolvers import (
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
)
from .resolvers.vrroomm import (
    VrrooommResolver,
    VrrooommResult,
    PassengerDelivery,
)

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


class ResolutionStatus(Enum):
    """Status of the resolution process."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AWAITING_INPUT = "awaiting_input"
    AREA_COMPLETE = "area_complete"
    ALL_COMPLETE = "all_complete"


@dataclass
class ResolutionContext:
    """Context for the current resolution state.

    Attributes:
        current_area: The action area currently being resolved.
        current_area_idx: Index into ACTION_RESOLUTION_ORDER.
        current_slot: The slot currently being resolved within the area.
        current_slot_idx: Index of the current slot in the area.
        status: Current status of the resolution.
        awaiting_player_id: Player who needs to make a decision (if any).
        valid_actions: List of valid actions for the current decision point.
    """

    current_area: ActionAreaType | None = None
    current_area_idx: int = 0
    current_slot: ActionSlot | None = None
    current_slot_idx: int = 0
    status: ResolutionStatus = ResolutionStatus.NOT_STARTED
    awaiting_player_id: int | None = None
    valid_actions: list[dict] = field(default_factory=list)


@dataclass
class AreaResolutionResult:
    """Result of resolving a single action area.

    Attributes:
        area_type: The action area that was resolved.
        resolved: Whether the area had markers to resolve.
        result: The area-specific result object.
    """

    area_type: ActionAreaType
    resolved: bool
    result: Any = None


@dataclass
class FullResolutionResult:
    """Result of resolving all action areas.

    Attributes:
        completed: Whether all areas were resolved.
        area_results: Results for each action area.
        game_ended: Whether the game ended during resolution.
        game_end_reason: Reason for game ending (if applicable).
    """

    completed: bool
    area_results: list[AreaResolutionResult] = field(default_factory=list)
    game_ended: bool = False
    game_end_reason: str | None = None


class ActionResolver:
    """Coordinates resolution of all action areas.

    The ActionResolver manages the resolution phase by:
    1. Iterating through action areas in the correct order
    2. Instantiating appropriate resolvers for each area
    3. Providing valid actions for player decisions
    4. Executing player choices and advancing state

    Usage (automatic resolution with default choices):
        resolver = ActionResolver(game_state)
        result = resolver.resolve_all()

    Usage (step-by-step with player choices):
        resolver = ActionResolver(game_state)
        resolver.start_resolution()

        while not resolver.is_complete():
            context = resolver.get_context()
            if context.status == ResolutionStatus.AWAITING_INPUT:
                actions = context.valid_actions
                chosen_action = player_chooses(actions)
                resolver.apply_action(chosen_action)
            else:
                resolver.advance()
    """

    def __init__(self, state: GameState):
        """Initialize the action resolver.

        Args:
            state: The current game state.
        """
        self.state = state
        self._context = ResolutionContext()
        self._area_results: list[AreaResolutionResult] = []

        # Current resolvers (created on demand)
        self._line_expansion_resolver: LineExpansionResolver | None = None
        self._buses_resolver: BusesResolver | None = None
        self._passengers_resolver: PassengersResolver | None = None
        self._buildings_resolver: BuildingsResolver | None = None
        self._time_clock_resolver: TimeClockResolver | None = None
        self._vrroomm_resolver: VrrooommResolver | None = None
        self._starting_player_resolver: StartingPlayerResolver | None = None

    def get_context(self) -> ResolutionContext:
        """Get the current resolution context."""
        return self._context

    def is_complete(self) -> bool:
        """Check if all resolution is complete."""
        return self._context.status == ResolutionStatus.ALL_COMPLETE

    def start_resolution(self) -> None:
        """Start the resolution process from the first area."""
        self._context.current_area_idx = 0
        self._context.status = ResolutionStatus.IN_PROGRESS
        self._advance_to_next_area_with_markers()

    def _advance_to_next_area_with_markers(self) -> None:
        """Move to the next action area that has markers to resolve.
        
        This method will iterate through action areas until it find one with markers 
        or it reaches the end of the resolution order. It handles automatic areas 
        and areas without player input iteratively to avoid recursion.
        """
        while self._context.current_area_idx < len(ACTION_RESOLUTION_ORDER):
            area_type = ACTION_RESOLUTION_ORDER[self._context.current_area_idx]
            markers = self.state.action_board.get_markers_to_resolve(area_type)

            # Time Clock is always visited for automatic clock advance
            is_mandatory_visit = (area_type == ActionAreaType.TIME_CLOCK)
            
            if markers or is_mandatory_visit:
                self._context.current_area = area_type
                self._context.current_slot_idx = 0
                self._context.current_slot = markers[0] if markers else None
                self._setup_area_resolver(area_type)
                
                # Check if this area needs player input
                self._check_for_player_input()
                
                # If no input needed (e.g. automatic Buses resolve, or empty Time Clock), 
                # advance automatically and continue the loop to the next area.
                if self._context.status != ResolutionStatus.AWAITING_INPUT:
                     # This will eventually call _finalize_area which increments current_area_idx
                     # We must ensures we don't recurse here.
                     # advance() -> _finalize_area() -> _advance_to_next_area_with_markers()
                     # To avoid recursion, we'll manually finalize the area and continue the loop.
                     self._resolve_area_automatically(area_type)
                     # After automatic resolution, we continue the loop to the next area
                     continue
                else:
                    # Found an area that needs player input, return to caller
                    return

            self._context.current_area_idx += 1

        # All areas processed
        self._context.status = ResolutionStatus.ALL_COMPLETE
        self._context.current_area = None
        self._context.current_slot = None

    def _resolve_area_automatically(self, area_type: ActionAreaType) -> None:
        """Resolve an area that requires no player input.
        
        This mimics the logic in advance() but without calling 
        _advance_to_next_area_with_markers at the end of _finalize_area.
        """
        if area_type == ActionAreaType.BUSES:
            self._resolve_buses_internal()
        elif area_type == ActionAreaType.STARTING_PLAYER:
            self._resolve_starting_player_internal()
        elif area_type == ActionAreaType.TIME_CLOCK:
            self._resolve_time_clock_auto_internal()
        else:
            # For other areas (Expansion, Passengers, Buildings, Vrroomm!)
            # If no input is needed, it means no markers were valid or something.
            # We just finalize the area normally.
            self._finalize_area_internal(area_type)

    def _finalize_area_internal(self, area_type: ActionAreaType, result: Any = None) -> None:
        """Internal version of _finalize_area that does NOT recurse."""
        area_result = AreaResolutionResult(
            area_type=area_type,
            resolved=True,
            result=result,
        )
        self._area_results.append(area_result)

        if area_type == ActionAreaType.BUILDINGS:
            if self._buildings_resolver and self._buildings_resolver.check_game_end_condition():
                self._context.status = ResolutionStatus.ALL_COMPLETE
                return

        if area_type == ActionAreaType.VRROOMM and self._vrroomm_resolver:
            self._vrroomm_resolver.clear_occupancy()

        self._context.current_area_idx += 1

    def _resolve_buses_internal(self) -> None:
        if not self._buses_resolver:
            self._buses_resolver = BusesResolver(self.state)
        result = self._buses_resolver.resolve()
        self._finalize_area_internal(ActionAreaType.BUSES, result)

    def _resolve_starting_player_internal(self) -> None:
        if not self._starting_player_resolver:
            self._starting_player_resolver = StartingPlayerResolver(self.state)
        result = self._starting_player_resolver.resolve()
        self._finalize_area_internal(ActionAreaType.STARTING_PLAYER, result)

    def _resolve_time_clock_auto_internal(self) -> None:
        if not self._time_clock_resolver:
            self._time_clock_resolver = TimeClockResolver(self.state)
        result = self._time_clock_resolver.resolve(TimeClockAction.ADVANCE_CLOCK)
        self._finalize_area_internal(ActionAreaType.TIME_CLOCK, result)
        if result.game_ended:
            self._context.status = ResolutionStatus.ALL_COMPLETE

    def _setup_area_resolver(self, area_type: ActionAreaType) -> None:
        """Create the resolver for a specific area."""
        if area_type == ActionAreaType.LINE_EXPANSION:
            self._line_expansion_resolver = LineExpansionResolver(self.state)
        elif area_type == ActionAreaType.BUSES:
            self._buses_resolver = BusesResolver(self.state)
        elif area_type == ActionAreaType.PASSENGERS:
            self._passengers_resolver = PassengersResolver(self.state)
        elif area_type == ActionAreaType.BUILDINGS:
            self._buildings_resolver = BuildingsResolver(self.state)
        elif area_type == ActionAreaType.TIME_CLOCK:
            self._time_clock_resolver = TimeClockResolver(self.state)
        elif area_type == ActionAreaType.VRROOMM:
            self._vrroomm_resolver = VrrooommResolver(self.state)
        elif area_type == ActionAreaType.STARTING_PLAYER:
            self._starting_player_resolver = StartingPlayerResolver(self.state)

    def _check_for_player_input(self) -> None:
        """Check if the current resolution point requires player input."""
        area_type = self._context.current_area

        if area_type is None:
            return

        valid_actions = self._get_valid_actions_for_area(area_type)

        if valid_actions:
            self._context.status = ResolutionStatus.AWAITING_INPUT
            self._context.valid_actions = valid_actions
            if valid_actions:
                self._context.awaiting_player_id = valid_actions[0].get("player_id")
        else:
            self._context.status = ResolutionStatus.IN_PROGRESS
            self._context.valid_actions = []
            self._context.awaiting_player_id = None

    def _get_valid_actions_for_area(self, area_type: ActionAreaType) -> list[dict]:
        """Get valid actions for the current point in an area's resolution."""
        if area_type == ActionAreaType.LINE_EXPANSION:
            if self._line_expansion_resolver:
                return self._line_expansion_resolver.get_valid_actions()
        elif area_type == ActionAreaType.PASSENGERS:
            if self._passengers_resolver:
                return self._passengers_resolver.get_valid_actions()
        elif area_type == ActionAreaType.BUILDINGS:
            if self._buildings_resolver:
                return self._buildings_resolver.get_valid_actions()
        elif area_type == ActionAreaType.TIME_CLOCK:
            if self._time_clock_resolver:
                return self._time_clock_resolver.get_valid_actions()
        elif area_type == ActionAreaType.VRROOMM:
            if self._vrroomm_resolver:
                return self._vrroomm_resolver.get_valid_actions()
        # BUSES and STARTING_PLAYER have no player choice
        return []

    def apply_action(self, action: dict) -> bool:
        """Apply a player's chosen action.

        Args:
            action: The action dictionary chosen by the player.

        Returns:
            True if action was applied successfully.

        Raises:
            ValueError: If no action is expected or action is invalid.
        """
        if self._context.status != ResolutionStatus.AWAITING_INPUT:
            raise ValueError("No player input expected at this time")

        area_type = self._context.current_area

        if area_type == ActionAreaType.LINE_EXPANSION:
            return self._apply_line_expansion_action(action)
        elif area_type == ActionAreaType.PASSENGERS:
            return self._apply_passengers_action(action)
        elif area_type == ActionAreaType.BUILDINGS:
            return self._apply_buildings_action(action)
        elif area_type == ActionAreaType.TIME_CLOCK:
            return self._apply_time_clock_action(action)
        elif area_type == ActionAreaType.VRROOMM:
            return self._apply_vrroomm_action(action)

        return False

    def _apply_line_expansion_action(self, action: dict) -> bool:
        """Apply a line expansion action."""
        if not self._line_expansion_resolver:
            return False

        placement = RailPlacement(
            edge_id=action["edge_id"],
            from_endpoint=action["from_endpoint"],
        )
        self._line_expansion_resolver.place_rail(placement)

        # Check if more placements needed
        if self._line_expansion_resolver.is_resolution_complete():
            self._finalize_area(ActionAreaType.LINE_EXPANSION)
        else:
            self._check_for_player_input()

        return True

    def _apply_passengers_action(self, action: dict) -> bool:
        """Apply a passengers action."""
        if not self._passengers_resolver:
            return False

        distribution = PassengerDistribution(distribution=action["distribution"])
        self._passengers_resolver.resolve_slot(distribution)

        if self._passengers_resolver.is_resolution_complete():
            self._finalize_area(ActionAreaType.PASSENGERS)
        else:
            self._check_for_player_input()

        return True

    def _apply_buildings_action(self, action: dict) -> bool:
        """Apply a buildings action."""
        if not self._buildings_resolver:
            return False

        placement = BuildingPlacement(
            node_id=action["node_id"],
            slot_index=action["slot_index"],
            building_type=action["building_type"],
        )
        self._buildings_resolver.place_building(placement)

        if self._buildings_resolver.is_resolution_complete():
            self._finalize_area(ActionAreaType.BUILDINGS)
        else:
            self._check_for_player_input()

        return True

    def _apply_time_clock_action(self, action: dict) -> bool:
        """Apply a time clock action."""
        if not self._time_clock_resolver:
            return False

        time_action = TimeClockAction(action["action"])
        result = self._time_clock_resolver.resolve(time_action)

        self._finalize_area(ActionAreaType.TIME_CLOCK, result)

        # Check for game end
        if result.game_ended:
            self._context.status = ResolutionStatus.ALL_COMPLETE

        return True

    def _apply_vrroomm_action(self, action: dict) -> bool:
        """Apply a vrroomm action."""
        if not self._vrroomm_resolver:
            return False

        delivery = PassengerDelivery(
            passenger_id=action["passenger_id"],
            from_node=action["from_node"],
            to_node=action["to_node"],
            building_slot_index=action["building_slot_index"],
        )
        self._vrroomm_resolver.deliver_passenger(delivery)

        # Check if more deliveries possible or slot is done
        remaining = self._vrroomm_resolver.get_deliveries_remaining_for_current_slot()
        actions = self._vrroomm_resolver.get_valid_actions()

        if remaining <= 0 or not actions:
            self._vrroomm_resolver.finalize_current_slot()
            if self._vrroomm_resolver.is_resolution_complete():
                self._finalize_area(ActionAreaType.VRROOMM)
            else:
                self._check_for_player_input()
        else:
            self._check_for_player_input()

        return True

    def skip_vrroomm_deliveries(self) -> None:
        """Skip remaining deliveries for the current Vrroomm! slot.

        Call this when the player chooses not to make more deliveries.
        """
        if self._vrroomm_resolver and self._context.current_area == ActionAreaType.VRROOMM:
            self._vrroomm_resolver.skip_remaining_deliveries()
            if self._vrroomm_resolver.is_resolution_complete():
                self._finalize_area(ActionAreaType.VRROOMM)
            else:
                self._check_for_player_input()

    def advance(self) -> None:
        """Advance resolution when no player input is needed.

        This resolves automatic actions (like Buses, Starting Player, Time Clock auto-advance)
        or skips slots/areas where no valid actions are available.
        """
        if self._context.status == ResolutionStatus.AWAITING_INPUT:
            raise ValueError("Cannot advance - player input required")

        area_type = self._context.current_area
        if area_type is None:
            return

        if area_type == ActionAreaType.BUSES:
            self._resolve_buses()
        elif area_type == ActionAreaType.STARTING_PLAYER:
            self._resolve_starting_player()
        elif area_type == ActionAreaType.TIME_CLOCK:
            self._resolve_time_clock_auto()
        elif area_type == ActionAreaType.VRROOMM:
            self._skip_current_slot_vrroomm()
        elif area_type == ActionAreaType.LINE_EXPANSION:
            self._skip_current_slot_line_expansion()
        elif area_type == ActionAreaType.PASSENGERS:
            self._skip_current_slot_passengers()
        elif area_type == ActionAreaType.BUILDINGS:
            self._skip_current_slot_buildings()
        else:
            # Fallback (should not happen for known areas)
            self._finalize_area(area_type)

    def _resolve_buses(self) -> None:
        """Resolve the Buses action area (no player choice)."""
        if not self._buses_resolver:
            self._buses_resolver = BusesResolver(self.state)

        result = self._buses_resolver.resolve()
        self._finalize_area(ActionAreaType.BUSES, result)

    def _resolve_starting_player(self) -> None:
        """Resolve the Starting Player action area (no player choice)."""
        if not self._starting_player_resolver:
            self._starting_player_resolver = StartingPlayerResolver(self.state)

        result = self._starting_player_resolver.resolve()
        self._finalize_area(ActionAreaType.STARTING_PLAYER, result)

    def _resolve_time_clock_auto(self) -> None:
        """Resolve Time Clock automatically (default: advance)."""
        if not self._time_clock_resolver:
            # Should not happen if _setup_area_resolver called
            self._time_clock_resolver = TimeClockResolver(self.state)

        # Call resolve() - it handles the "no markers" case or "advance" fallback
        result = self._time_clock_resolver.resolve(TimeClockAction.ADVANCE_CLOCK)
        self._finalize_area(ActionAreaType.TIME_CLOCK, result)

        if result.game_ended:
            self._context.status = ResolutionStatus.ALL_COMPLETE

    def _skip_current_slot_vrroomm(self) -> None:
        """Skip current slot in Vrroomm! (if no valid actions)."""
        if self._vrroomm_resolver:
            self._vrroomm_resolver.finalize_current_slot()
            
            if self._vrroomm_resolver.is_resolution_complete():
                self._finalize_area(ActionAreaType.VRROOMM)
            else:
                self._check_for_player_input()

    def _skip_current_slot_line_expansion(self) -> None:
        """Skip current slot in Line Expansion (if no valid actions)."""
        if self._line_expansion_resolver:
            # LineExpansionResolver uses internal index, we need to finalize it
            # The resolver's _finalize_current_slot is used internally.
            # We can force it by calling resolve_all with empty list? No, that resolves ALL.
            # We need to expose a skip/finalize method or simulate it.
            # Actually, `place_rail` is what we normally call.
            # If we are here, it means NO PLACE_RAIL actions were valid.
            # So we should force finalize the current slot.
            # The LineExpansionResolver has `_finalize_current_slot` but it might not be public.
            # Let's rely on `_finalize_current_slot` being accessible or add a wrapper.
            # Since I can't easily edit resolver API in this step without re-reading, 
            # I will access the protected method _finalize_current_slot() which exists.
            self._line_expansion_resolver._finalize_current_slot()
            
            if self._line_expansion_resolver.is_resolution_complete():
                self._finalize_area(ActionAreaType.LINE_EXPANSION)
            else:
                self._check_for_player_input()

    def _skip_current_slot_passengers(self) -> None:
        """Skip current slot in Passengers (if no valid actions)."""
        if self._passengers_resolver:
             # Passengers logic: advance index manually if no public skip method
             self._passengers_resolver._current_slot_idx += 1
             
             if self._passengers_resolver.is_resolution_complete():
                 self._finalize_area(ActionAreaType.PASSENGERS)
             else:
                 self._check_for_player_input()

    def _skip_current_slot_buildings(self) -> None:
        """Skip current slot in Buildings (if no valid actions)."""
        if self._buildings_resolver:
             # Buildings logic: advance index manually
             self._buildings_resolver._current_slot_idx += 1
             
             if self._buildings_resolver.is_resolution_complete():
                 self._finalize_area(ActionAreaType.BUILDINGS)
             else:
                 self._check_for_player_input()

    def _finalize_area(self, area_type: ActionAreaType, result: Any = None) -> None:
        """Finalize an area's resolution and move to the next."""
        area_result = AreaResolutionResult(
            area_type=area_type,
            resolved=True,
            result=result,
        )
        self._area_results.append(area_result)

        # Check for game end conditions
        if area_type == ActionAreaType.BUILDINGS:
            if self._buildings_resolver and self._buildings_resolver.check_game_end_condition():
                self._context.status = ResolutionStatus.ALL_COMPLETE
                return

        # Move to next area
        if area_type == ActionAreaType.VRROOMM and self._vrroomm_resolver:
            self._vrroomm_resolver.clear_occupancy()

        self._context.current_area_idx += 1
        self._advance_to_next_area_with_markers()

    def resolve_all(self) -> FullResolutionResult:
        """Resolve all action areas using default choices.
        
        This method resolves everything automatically, making default
        choices for any player decisions. Useful for training and
        simulations.
        """
        self._area_results = []
        game_ended = False
        game_end_reason = None

        for area_type in ACTION_RESOLUTION_ORDER:
            markers = self.state.action_board.get_markers_to_resolve(area_type)
            
            # Time Clock is always resolved, even without markers (automatic advance)
            is_mandatory_visit = (area_type == ActionAreaType.TIME_CLOCK)
            
            if not markers and not is_mandatory_visit:
                continue

            result = self._resolve_area_with_defaults(area_type)

            self._area_results.append(AreaResolutionResult(
                area_type=area_type,
                resolved=True,
                result=result,
            ))

            # Check for game end conditions
            if area_type == ActionAreaType.TIME_CLOCK:
                if hasattr(result, 'game_ended') and result.game_ended:
                    game_ended = True
                    game_end_reason = "Last time stone taken"
                    break

            if area_type == ActionAreaType.BUILDINGS:
                if hasattr(result, 'all_slots_filled') and result.all_slots_filled:
                    game_ended = True
                    game_end_reason = "All building slots filled"
                    break

        return FullResolutionResult(
            completed=True,
            area_results=self._area_results,
            game_ended=game_ended,
            game_end_reason=game_end_reason,
        )

    def _resolve_area_with_defaults(self, area_type: ActionAreaType) -> Any:
        """Resolve a single area using default choices.

        Args:
            area_type: The action area to resolve.

        Returns:
            The area-specific result object.
        """
        if area_type == ActionAreaType.LINE_EXPANSION:
            resolver = LineExpansionResolver(self.state)
            return resolver.resolve_all()

        elif area_type == ActionAreaType.BUSES:
            resolver = BusesResolver(self.state)
            return resolver.resolve()

        elif area_type == ActionAreaType.PASSENGERS:
            resolver = PassengersResolver(self.state)
            return resolver.resolve_all()

        elif area_type == ActionAreaType.BUILDINGS:
            resolver = BuildingsResolver(self.state)
            return resolver.resolve_all()

        elif area_type == ActionAreaType.TIME_CLOCK:
            resolver = TimeClockResolver(self.state)
            # Default: advance clock (not stop)
            return resolver.resolve(TimeClockAction.ADVANCE_CLOCK)

        elif area_type == ActionAreaType.VRROOMM:
            resolver = VrrooommResolver(self.state)
            return resolver.resolve_all()

        elif area_type == ActionAreaType.STARTING_PLAYER:
            resolver = StartingPlayerResolver(self.state)
            return resolver.resolve()

        return None

    def get_results(self) -> list[AreaResolutionResult]:
        """Get the results from all resolved areas."""
        return list(self._area_results)
