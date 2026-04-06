"""Vrroomm two-stage selection state for hierarchical RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.game_state import GameState


@dataclass
class VrroommStageState:
    """Tracks Vrroomm two-stage selection.

    Stages:
        0 = not in Vrroomm resolution
        1 = selecting passenger
        2 = selecting destination
    """

    stage: int = 0
    selected_passenger_id: Optional[int] = None

    def enter(self) -> None:
        """Enter Vrroomm resolution (stage 1)."""
        self.stage = 1
        self.selected_passenger_id = None

    def select_passenger(self, passenger_id: int) -> None:
        """Select a passenger and advance to stage 2."""
        if self.stage != 1:
            raise RuntimeError("VrroommStageState: select_passenger called in wrong stage")
        self.selected_passenger_id = int(passenger_id)
        self.stage = 2

    def build_delivery_action(
        self,
        state: GameState,
        to_node: int,
        building_slot_index: int,
    ) -> dict:
        """Build a resolver action dict for the selected passenger.

        After building the action, returns to stage 1 (ready for next passenger).
        Caller should invoke `complete()` to exit Vrroomm when the slot is done.
        """
        if self.stage != 2 or self.selected_passenger_id is None:
            raise RuntimeError("VrroommStageState: build_delivery_action called in wrong stage")

        passenger = state.passenger_manager.get_passenger(self.selected_passenger_id)
        if passenger is None:
            raise ValueError(f"Passenger {self.selected_passenger_id} not found")

        action = {
            "passenger_id": self.selected_passenger_id,
            "from_node": passenger.location,
            "to_node": int(to_node),
            "building_slot_index": int(building_slot_index),
        }

        # Reset to stage 1 for potential additional deliveries
        self.stage = 1
        self.selected_passenger_id = None
        return action

    def skip(self) -> None:
        """Skip remaining deliveries and exit Vrroomm (stage 0)."""
        self.stage = 0
        self.selected_passenger_id = None

    def complete(self) -> None:
        """Exit Vrroomm after the slot is fully resolved."""
        self.stage = 0
        self.selected_passenger_id = None
