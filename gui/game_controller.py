"""Game controller for integrating the game engine with the GUI."""

from __future__ import annotations

from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum, auto

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from core.constants import Phase, ActionAreaType, ACTION_RESOLUTION_ORDER, MIN_MARKERS_PER_ROUND
from core.game_state import GameState
from core.board import NodeId, EdgeId

from engine.game_engine import GameEngine, Action, ActionType
from engine.action_resolver import ActionResolver, ResolutionStatus

from gui.main_window import MainWindow
from gui.gui_renderer import GUIRenderer
from gui.gui_prompter import GUIPrompter
from gui.dialogs import (
    BuildingPlacementDialog, PassengerDistributionDialog,
    VrrooommDialog, TimeClockDialog, GameOverDialog
)


class InputMode(Enum):
    """What kind of input the controller is waiting for."""
    NONE = auto()
    BUILDING_PLACEMENT = auto()
    RAIL_PLACEMENT = auto()
    ACTION_SELECTION = auto()
    RESOLUTION_BUILDING = auto()
    RESOLUTION_RAIL = auto()
    PASSENGER_DISTRIBUTION = auto()
    VRROOMM_DELIVERY = auto()


class GameController(QObject):
    """Controller that manages the game loop and GUI interaction.

    This class orchestrates:
    - Game engine operations
    - GUI updates
    - Player input handling via direct board clicks
    - Action resolution

    Unlike the CLI driver which blocks for input, this controller
    uses Qt's event loop and signals for asynchronous operation.
    """

    game_started = Signal()
    game_ended = Signal()
    turn_started = Signal(int)  # player_id
    phase_changed = Signal(str)  # phase name

    def __init__(self, main_window: MainWindow):
        super().__init__()

        self._window = main_window
        self._engine = GameEngine()
        self._renderer = GUIRenderer(main_window)
        self._prompter = GUIPrompter(main_window)

        self._action_resolver: Optional[ActionResolver] = None
        self._game_active = False

        # Input mode tracking
        self._input_mode = InputMode.NONE
        self._valid_actions: list[Action] = []
        self._valid_action_map: dict = {}  # Maps click targets to actions
        self._resolution_valid_actions: list[dict] = []
        self._distribution_current: dict[int, int] = {}  # For PASSENGER_DISTRIBUTION
        self._vrroomm_selected_passenger: Optional[int] = None  # For VRROOMM_DELIVERY

        # Connect window signals
        main_window.node_clicked.connect(self._on_node_clicked)
        main_window.edge_clicked.connect(self._on_edge_clicked)
        main_window.building_slot_clicked.connect(self._on_building_slot_clicked)
        main_window.action_board_clicked.connect(self._on_action_board_clicked)
        main_window.pass_clicked.connect(self._on_pass_clicked)
        main_window.set_new_game_callback(self.start_new_game)

    def start_new_game(self, num_players: int) -> None:
        """Start a new game with the specified number of players."""
        self._engine.reset(num_players=num_players)
        self._game_active = True
        self._action_resolver = None
        self._input_mode = InputMode.NONE

        self._renderer.render_message(f"New game started with {num_players} players!")
        self._renderer.render_state(self._engine.state)

        self.game_started.emit()
        self._process_game_state()

    def _process_game_state(self) -> None:
        """Process the current game state and prompt for action if needed."""
        if not self._game_active:
            return

        state = self._engine.state

        if state.phase == Phase.GAME_OVER:
            self._handle_game_over()
            return

        # Get valid actions and update display
        self._valid_actions = self._engine.get_valid_actions()

        if state.phase == Phase.SETUP_BUILDINGS:
            self._handle_setup_buildings()
        elif state.phase == Phase.SETUP_RAILS_FORWARD:
            self._handle_setup_rails("Click on a highlighted edge to place your first rail segment")
        elif state.phase == Phase.SETUP_RAILS_REVERSE:
            self._handle_setup_rails("Click on a highlighted edge to extend your rail network")
        elif state.phase == Phase.CHOOSING_ACTIONS:
            self._handle_choosing_actions()
        elif state.phase == Phase.RESOLVING_ACTIONS:
            self._handle_resolving_actions()
        elif state.phase == Phase.CLEANUP:
            from gui.dialogs import CleanupDialog
            dialog = CleanupDialog(self._window)
            if dialog.exec():
                self._engine.resolve_cleanup()
                self._renderer.render_state(self._engine.state)
                QTimer.singleShot(100, self._process_game_state)
            else:
                # If they cancel? We probably shouldn't let them easily, but let's just show it again
                QTimer.singleShot(100, self._process_game_state)

    def _handle_setup_buildings(self) -> None:
        """Handle the setup buildings phase - direct click mode."""
        if not self._valid_actions:
            return

        current_player = self._engine.get_current_player()
        self._renderer.render_message(
            f"Player {current_player.player_id}: Click on a highlighted node to place a building (Zone A)"
        )

        # Build map of (node_id, slot_index) -> list of actions
        self._valid_action_map = {}
        valid_slots = set()
        for action in self._valid_actions:
            node_id = action.params["node_id"]
            slot_idx = action.params["slot_index"]
            valid_slots.add((node_id, slot_idx))
            if (node_id, slot_idx) not in self._valid_action_map:
                self._valid_action_map[(node_id, slot_idx)] = []
            self._valid_action_map[(node_id, slot_idx)].append(action)

        # Highlight valid slots
        self._window.highlight_valid_slots(valid_slots)
        self._input_mode = InputMode.BUILDING_PLACEMENT

    def _handle_setup_rails(self, message: str) -> None:
        """Handle the setup rails phases - direct click mode."""
        if not self._valid_actions:
            return

        current_player = self._engine.get_current_player()
        self._renderer.render_message(f"Player {current_player.player_id}: {message}")

        # Build map of edge_id -> action
        self._valid_action_map = {}
        valid_edges = set()
        for action in self._valid_actions:
            edge_id = tuple(action.params["edge_id"])
            valid_edges.add(edge_id)
            self._valid_action_map[edge_id] = action

        # Highlight valid edges
        self._window.highlight_valid_edges(valid_edges)
        self._input_mode = InputMode.RAIL_PLACEMENT

    def _handle_choosing_actions(self) -> None:
        """Handle the choosing actions phase - click on action board."""
        if not self._valid_actions:
            return

        current_player = self._engine.get_current_player()
        markers_placed = current_player.markers_placed_this_round
        can_pass = markers_placed >= MIN_MARKERS_PER_ROUND or current_player.action_markers_remaining == 0

        self._renderer.render_message(
            f"Player {current_player.player_id}: Click on a highlighted action area "
            f"(Markers: {current_player.action_markers_remaining}, Placed this round: {markers_placed})"
        )

        # Build map of area_type -> action
        self._valid_action_map = {}
        available_areas = []
        has_pass = False

        for action in self._valid_actions:
            if action.action_type == ActionType.PASS:
                has_pass = True
                self._valid_action_map["PASS"] = action
            elif action.action_type == ActionType.PLACE_MARKER:
                area_type_str = action.params["area_type"]
                try:
                    area_type = ActionAreaType(area_type_str)
                    if area_type not in available_areas:
                        available_areas.append(area_type)
                    self._valid_action_map[area_type_str] = action
                except ValueError:
                    pass

        # Highlight available action areas and show pass button
        self._window.highlight_valid_action_areas(available_areas)
        self._window.set_pass_button_state(True, can_pass and has_pass)
        self._input_mode = InputMode.ACTION_SELECTION

    def _handle_resolving_actions(self) -> None:
        """Handle the resolution phase."""
        if self._action_resolver is None:
            self._action_resolver = ActionResolver(self._engine.state)
            self._action_resolver.start_resolution()

        if self._action_resolver.is_complete():
            self._finish_resolution()
            return

        context = self._action_resolver.get_context()

        if context.status == ResolutionStatus.AWAITING_INPUT:
            self._handle_resolution_input(context)
        else:
            # Automatic resolution
            self._action_resolver.advance()
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)

    def _handle_resolution_input(self, context) -> None:
        """Handle player input during resolution."""
        area_type = context.current_area
        valid_actions = context.valid_actions

        if not valid_actions:
            self._action_resolver.advance()
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
            return

        player_id = valid_actions[0].get("player_id", 0)
        self._renderer.render_message(f"{area_type.value.replace('_', ' ').upper()} - Player {player_id}")
        self._resolution_valid_actions = valid_actions

        if area_type == ActionAreaType.LINE_EXPANSION:
            self._handle_line_expansion_input(valid_actions)
        elif area_type == ActionAreaType.PASSENGERS:
            self._handle_passengers_input(valid_actions)
        elif area_type == ActionAreaType.BUILDINGS:
            self._handle_buildings_input(valid_actions)
        elif area_type == ActionAreaType.TIME_CLOCK:
            self._handle_time_clock_input(valid_actions)
        elif area_type == ActionAreaType.VRROOMM:
            self._handle_vrroomm_input(valid_actions)
        else:
            # No input needed
            self._action_resolver.advance()
            QTimer.singleShot(100, self._process_game_state)

    def _handle_line_expansion_input(self, valid_actions: list[dict]) -> None:
        """Handle line expansion - click on edge."""
        if not valid_actions:
            return

        player_id = valid_actions[0].get("player_id", 0)
        # Build map of edge_id -> action
        self._valid_action_map = {}
        valid_edges = set()
        for action in valid_actions:
            edge_id = tuple(action["edge_id"])
            valid_edges.add(edge_id)
            self._valid_action_map[edge_id] = action

        self._window.highlight_valid_edges(valid_edges)
        
        # Get progress info
        count = 0
        total = 0
        if self._action_resolver and self._action_resolver._line_expansion_resolver:
            total = self._action_resolver._line_expansion_resolver.get_segments_remaining_for_current_slot()
            current_slot = self._action_resolver._line_expansion_resolver.get_current_slot()
            if current_slot:
                total_for_slot = self._action_resolver._line_expansion_resolver.get_segments_for_slot(current_slot)
                placed = total_for_slot - total
                count = placed + 1
                total = total_for_slot

        self._renderer.render_message(
            f"LINE EXPANSION: Placing rail {count} of {total} for Player {player_id}"
        )
        self._input_mode = InputMode.RESOLUTION_RAIL

    def _handle_passengers_input(self, valid_actions: list[dict]) -> None:
        """Handle passengers distribution choice - interactive."""
        if not valid_actions:
            self._action_resolver.advance()
            QTimer.singleShot(100, self._process_game_state)
            return

        player_id = valid_actions[0].get("player_id", 0)
        # Find distribution options
        first_action = valid_actions[0]
        distribution = first_action.get("distribution", {})
        station_ids = list(distribution.keys())
        total = sum(distribution.values())

        self._distribution_current = {sid: 0 for sid in station_ids}
        self._distribution_total = total
        self._resolution_valid_actions = valid_actions

        # Highlight train stations
        self._window.highlight_valid_nodes(set(station_ids))
        self._window.set_distribution_preview(self._distribution_current)
        self._window.set_pass_button_state(True, False, f"Confirm (Distributed: 0/{total})")
        
        self._renderer.render_message(
            f"PASSENGERS: Distribute {total} passengers for Player {player_id}"
        )
        self._input_mode = InputMode.PASSENGER_DISTRIBUTION

    def _handle_buildings_input(self, valid_actions: list[dict]) -> None:
        """Handle buildings placement - click on node."""
        if not valid_actions:
            return

        player_id = valid_actions[0].get("player_id", 0)
        # Build map of (node_id, slot_index) -> list of actions
        self._valid_action_map = {}
        valid_slots = set()
        for action in valid_actions:
            node_id = action["node_id"]
            slot_idx = action["slot_index"]
            valid_slots.add((node_id, slot_idx))
            if (node_id, slot_idx) not in self._valid_action_map:
                self._valid_action_map[(node_id, slot_idx)] = []
            self._valid_action_map[(node_id, slot_idx)].append(action)

        self._window.highlight_valid_slots(valid_slots)
        
        # Get progress info
        count = 0
        total = 0
        if self._action_resolver and self._action_resolver._buildings_resolver:
            remaining = self._action_resolver._buildings_resolver.get_buildings_remaining_for_current_slot()
            current_slot = self._action_resolver._buildings_resolver.get_current_slot()
            if current_slot:
                total_for_slot = self._action_resolver._buildings_resolver.get_buildings_for_slot(current_slot)
                count = (total_for_slot - remaining) + 1
                total = total_for_slot

        self._renderer.render_message(
            f"BUILDINGS: Placing building {count} of {total} for Player {player_id}"
        )
        self._input_mode = InputMode.RESOLUTION_BUILDING

    def _handle_time_clock_input(self, valid_actions: list[dict]) -> None:
        """Handle time clock choice - use dialog."""
        current_pos = self._engine.state.global_state.time_clock_position.value
        stones = self._engine.state.global_state.time_stones_remaining

        dialog = TimeClockDialog(current_pos, stones, self._window)
        if dialog.exec():
            choice = dialog.get_choice()
            if choice:
                for action in valid_actions:
                    action_type = action.get("action")
                    if hasattr(action_type, 'value'):
                        action_val = action_type.value
                    else:
                        action_val = action_type
                    if action_val == choice:
                        self._action_resolver.apply_action(action)
                        break

                self._renderer.render_state(self._engine.state)
                QTimer.singleShot(100, self._process_game_state)
        else:
            self._action_resolver.advance()
            QTimer.singleShot(100, self._process_game_state)

    def _handle_vrroomm_input(self, valid_actions: list[dict]) -> None:
        """Handle Vrroomm! delivery choice - interactive."""
        if not valid_actions:
            self._action_resolver.skip_vrroomm_deliveries()
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
            return

        player_id = valid_actions[0].get("player_id", 0)
        self._resolution_valid_actions = valid_actions
        self._vrroomm_selected_passenger = None
        
        # Build map of from_node -> building_slot_index -> list of (to_node, action)
        self._valid_action_map = {}
        from_nodes = set()
        for action in valid_actions:
            fn = action["from_node"]
            from_nodes.add(fn)
            if fn not in self._valid_action_map:
                self._valid_action_map[fn] = {}
            
            slot_idx = action["building_slot_index"]
            if slot_idx not in self._valid_action_map[fn]:
                self._valid_action_map[fn][slot_idx] = []
            
            self._valid_action_map[fn][slot_idx].append(action)

        self._window.highlight_valid_nodes(from_nodes)
        self._window.set_pass_button_state(True, True, "SKIP Remaining Deliveries")
        
        # Progress info
        remaining = 0
        total = 0
        if self._action_resolver and self._action_resolver._vrroomm_resolver:
            remaining = self._action_resolver._vrroomm_resolver.get_deliveries_remaining_for_current_slot()
            player = self._engine.state.get_player(player_id)
            total = player.buses
        
        delivered = total - remaining
        self._renderer.render_message(
            f"VRROOMM!: Delivery {delivered + 1} of up to {total} for Player {player_id}"
        )
        self._input_mode = InputMode.VRROOMM_DELIVERY

    def _finish_resolution(self) -> None:
        """Finish the resolution phase."""
        self._engine.state.global_state.current_resolution_area_idx = len(ACTION_RESOLUTION_ORDER)
        self._action_resolver = None
        self._input_mode = InputMode.NONE

        self._engine._check_phase_transition()
        self._renderer.render_state(self._engine.state)
        QTimer.singleShot(100, self._process_game_state)

    def _handle_game_over(self) -> None:
        """Handle the game over state."""
        self._game_active = False
        self._input_mode = InputMode.NONE
        self._renderer.render_message("GAME OVER!")
        self._renderer.render_state(self._engine.state)

        dialog = GameOverDialog(self._engine.state, self._window)
        dialog.exec()

        self.game_ended.emit()

    def _on_node_clicked(self, node_id: int) -> None:
        """Handle node click from board widget."""
        if self._input_mode == InputMode.BUILDING_PLACEMENT:
            # Check if this node has only one valid slot, and use it
            valid_slots_for_node = [s for s in self._valid_action_map.keys() if isinstance(s, tuple) and s[0] == node_id]
            if len(valid_slots_for_node) == 1:
                self._handle_building_slot_click(valid_slots_for_node[0][0], valid_slots_for_node[0][1])
            elif len(valid_slots_for_node) > 1:
                self._renderer.render_message("Please click on a specific building slot.")
        elif self._input_mode == InputMode.RESOLUTION_BUILDING:
            valid_slots_for_node = [s for s in self._valid_action_map.keys() if isinstance(s, tuple) and s[0] == node_id]
            if len(valid_slots_for_node) == 1:
                self._handle_resolution_slot_click(valid_slots_for_node[0][0], valid_slots_for_node[0][1])
            elif len(valid_slots_for_node) > 1:
                self._renderer.render_message("Please click on a specific building slot.")
        elif self._input_mode == InputMode.VRROOMM_DELIVERY:
            self._handle_vrroomm_node_click(node_id)
        elif self._input_mode == InputMode.PASSENGER_DISTRIBUTION:
            self._handle_passenger_node_click(node_id)

    def _on_building_slot_clicked(self, node_id: int, slot_index: int) -> None:
        """Handle building slot click from board widget."""
        if self._input_mode == InputMode.BUILDING_PLACEMENT:
            self._handle_building_slot_click(node_id, slot_index)
        elif self._input_mode == InputMode.RESOLUTION_BUILDING:
            self._handle_resolution_slot_click(node_id, slot_index)

    def _handle_building_slot_click(self, node_id: int, slot_index: int) -> None:
        """Handle slot click during setup building placement."""
        key = (node_id, slot_index)
        if key not in self._valid_action_map:
            return

        actions = self._valid_action_map[key]
        
        if len(actions) == 1:
            action = actions[0]
            result = self._engine.step(action)
            if result.success:
                self._window.clear_highlights()
                self._input_mode = InputMode.NONE
                self._renderer.render_state(self._engine.state)
                QTimer.singleShot(100, self._process_game_state)
            else:
                self._renderer.render_error(result.info.get("error", "Unknown error"))
        else:
            # Multiple building types - show dialog
            placements = []
            for action in actions:
                node = self._engine.state.board.get_node(action.params["node_id"])
                slot_idx = action.params["slot_index"]
                placements.append({
                    "node_id": action.params["node_id"],
                    "slot_index": slot_idx,
                    "building_type": action.params["building_type"],
                    "zone": node.building_slots[slot_idx].zone.value,
                    "action": action
                })

            dialog = BuildingPlacementDialog(placements, self._window)
            if dialog.exec():
                selected = dialog.get_selected()
                if selected:
                    result = self._engine.step(selected["action"])
                    if result.success:
                        self._window.clear_highlights()
                        self._input_mode = InputMode.NONE
                        self._renderer.render_state(self._engine.state)
                        QTimer.singleShot(100, self._process_game_state)
                    else:
                        self._renderer.render_error(result.info.get("error", "Unknown error"))

    def _handle_resolution_slot_click(self, node_id: int, slot_index: int) -> None:
        """Handle slot click during resolution building placement."""
        key = (node_id, slot_index)
        if key not in self._valid_action_map:
            return

        actions = self._valid_action_map[key]
        if len(actions) == 1:
            action = actions[0]
            self._action_resolver.apply_action(action)
            self._window.clear_highlights()
            self._input_mode = InputMode.NONE
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
        else:
            # Multiple building types for this slot - show dialog
            placements = []
            for action in actions:
                node = self._engine.state.board.get_node(action["node_id"])
                slot_idx = action["slot_index"]
                placement = action.copy()
                placement["zone"] = node.building_slots[slot_idx].zone.value
                placements.append(placement)

            dialog = BuildingPlacementDialog(placements, self._window)
            if dialog.exec():
                selected = dialog.get_selected()
                if selected:
                    self._action_resolver.apply_action(selected)
                    self._window.clear_highlights()
                    self._input_mode = InputMode.NONE
                    self._renderer.render_state(self._engine.state)
                    QTimer.singleShot(100, self._process_game_state)

    def _handle_vrroomm_node_click(self, node_id: int) -> None:
        """Handle node click during Vrroomm! delivery."""
        if self._vrroomm_selected_passenger is None:
            # Selecting source node
            if node_id not in self._valid_action_map:
                return
            
            self._vrroomm_selected_passenger = node_id
            # Highlight destination nodes
            destinations = set()
            for slot_idx in self._valid_action_map[node_id]:
                for action in self._valid_action_map[node_id][slot_idx]:
                    destinations.add(action["to_node"])
            
            self._window.clear_highlights()
            self._window.highlight_valid_nodes(destinations)
            self._renderer.render_message(f"VRROOMM!: Click on a destination node for the passenger from Node {node_id}. Click this node again to deselect.")
        else:
            # If clicked the same node, deselect it
            if node_id == self._vrroomm_selected_passenger:
                self._vrroomm_selected_passenger = None
                self._window.clear_highlights()
                self._window.highlight_valid_nodes(set(self._valid_action_map.keys()))
                self._renderer.render_message("VRROOMM!: Select a passenger (blue circle on node) to deliver.")
                return

            # If clicked another valid source node, switch to it
            if node_id in self._valid_action_map:
                self._vrroomm_selected_passenger = None
                self._handle_vrroomm_node_click(node_id)
                return

            # Selecting destination node
            from_node = self._vrroomm_selected_passenger
            possible_actions = []
            for slot_idx in self._valid_action_map[from_node]:
                for action in self._valid_action_map[from_node][slot_idx]:
                    if action["to_node"] == node_id:
                        possible_actions.append(action)
            
            if not possible_actions:
                # We already handled re-selecting or switching source nodes above.
                # If we reach here, it's an invalid destination click.
                return

            if len(possible_actions) == 1:
                try:
                    self._action_resolver.apply_action(possible_actions[0])
                    self._window.clear_highlights()
                    self._input_mode = InputMode.NONE
                    self._renderer.render_state(self._engine.state)
                    QTimer.singleShot(100, self._process_game_state)
                except ValueError as e:
                    self._renderer.render_error(str(e))
            else:
                # Multiple slots for same destination - show dialog
                dialog = VrrooommDialog(possible_actions, 1, 99, self._window)
                if dialog.exec():
                    if not dialog.is_skip():
                        selected = dialog.get_selected()
                        if selected:
                            try:
                                self._action_resolver.apply_action(selected)
                            except ValueError as e:
                                self._renderer.render_error(str(e))
                
                self._window.clear_highlights()
                self._input_mode = InputMode.NONE
                self._renderer.render_state(self._engine.state)
                QTimer.singleShot(100, self._process_game_state)

    def _handle_passenger_node_click(self, node_id: int) -> None:
        """Handle node click during passenger distribution."""
        if node_id not in self._distribution_current:
            return
        
        current_sum = sum(self._distribution_current.values())
        if current_sum < self._distribution_total:
            self._distribution_current[node_id] += 1
            new_sum = current_sum + 1
            can_confirm = (new_sum == self._distribution_total)
            self._window.set_distribution_preview(self._distribution_current)
            self._window.set_pass_button_state(
                True, can_confirm, f"Confirm (Distributed: {new_sum}/{self._distribution_total})"
            )
        else:
            # Reset this station to 0 to allow redistribution
            self._distribution_current[node_id] = 0
            new_sum = sum(self._distribution_current.values())
            self._window.set_distribution_preview(self._distribution_current)
            self._window.set_pass_button_state(
                True, False, f"Confirm (Distributed: {new_sum}/{self._distribution_total})"
            )

    def _on_edge_clicked(self, edge_id: tuple) -> None:
        """Handle edge click from board widget."""
        if self._input_mode == InputMode.RAIL_PLACEMENT:
            self._handle_rail_edge_click(edge_id)
        elif self._input_mode == InputMode.RESOLUTION_RAIL:
            self._handle_resolution_rail_click(edge_id)

    def _handle_rail_edge_click(self, edge_id: tuple) -> None:
        """Handle edge click during setup rail placement."""
        # Normalize edge_id
        edge_id = (min(edge_id), max(edge_id))

        if edge_id not in self._valid_action_map:
            self._renderer.render_error(f"Edge {edge_id} is not a valid placement target")
            return

        action = self._valid_action_map[edge_id]
        result = self._engine.step(action)
        if result.success:
            self._window.clear_highlights()
            self._input_mode = InputMode.NONE
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
        else:
            self._renderer.render_error(result.info.get("error", "Unknown error"))

    def _handle_resolution_rail_click(self, edge_id: tuple) -> None:
        """Handle edge click during resolution rail placement."""
        edge_id = (min(edge_id), max(edge_id))

        if edge_id not in self._valid_action_map:
            self._renderer.render_error(f"Edge {edge_id} is not a valid placement target")
            return

        action = self._valid_action_map[edge_id]
        self._action_resolver.apply_action(action)
        self._window.clear_highlights()
        self._input_mode = InputMode.NONE
        self._renderer.render_state(self._engine.state)
        QTimer.singleShot(100, self._process_game_state)

    def _on_action_board_clicked(self, area_type: str, slot_label: str) -> None:
        """Handle action board slot click."""
        if self._input_mode != InputMode.ACTION_SELECTION:
            return

        if area_type not in self._valid_action_map:
            self._renderer.render_error(f"Cannot place marker in {area_type}")
            return

        action = self._valid_action_map[area_type]
        result = self._engine.step(action)
        if result.success:
            self._window.clear_highlights()
            self._input_mode = InputMode.NONE
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
        else:
            self._renderer.render_error(result.info.get("error", "Unknown error"))

    def _on_pass_clicked(self) -> None:
        """Handle Pass button click."""
        if self._input_mode == InputMode.ACTION_SELECTION:
            if "PASS" not in self._valid_action_map:
                self._renderer.render_error("Cannot pass at this time")
                return

            action = self._valid_action_map["PASS"]
            result = self._engine.step(action)
            if result.success:
                self._window.clear_highlights()
                self._input_mode = InputMode.NONE
                self._renderer.render_state(self._engine.state)
                QTimer.singleShot(100, self._process_game_state)
            else:
                self._renderer.render_error(result.info.get("error", "Unknown error"))
        elif self._input_mode == InputMode.PASSENGER_DISTRIBUTION:
            # Confirm distribution
            dist = self._distribution_current
            # Find matching action
            for action in self._resolution_valid_actions:
                if action.get("distribution") == dist:
                    self._action_resolver.apply_action(action)
                    break
            else:
                # Create action if needed (should be valid)
                action = self._resolution_valid_actions[0].copy()
                action["distribution"] = dist
                self._action_resolver.apply_action(action)

            self._window.clear_highlights()
            self._input_mode = InputMode.NONE
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)
        elif self._input_mode == InputMode.VRROOMM_DELIVERY:
            # Skip remaining deliveries
            self._action_resolver.skip_vrroomm_deliveries()
            self._window.clear_highlights()
            self._input_mode = InputMode.NONE
            self._renderer.render_state(self._engine.state)
            QTimer.singleShot(100, self._process_game_state)

    @property
    def state(self) -> Optional[GameState]:
        """Get the current game state."""
        return self._engine.state if self._game_active else None

    @property
    def is_game_active(self) -> bool:
        """Check if a game is currently active."""
        return self._game_active
