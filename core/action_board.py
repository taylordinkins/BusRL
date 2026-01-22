"""Action board model for the Bus game engine.

The action board contains 7 action areas where players place markers
during the Choosing Actions phase. Markers are resolved in a fixed order
during the Resolving Actions phase.

Slot labeling and layout:
- Slots are always labeled A, B, C, D, E, F
- Markers are always placed A first, then B, then C, etc.
- Physical layout (left-to-right) varies by area:
  - LINE_EXPANSION, BUILDINGS: F-E-D-C-B-A (A on right)
  - PASSENGERS, VRROOMM: A-B-C-D-E-F (A on left)
  - Single-slot areas (BUSES, TIME_CLOCK, STARTING_PLAYER): just A
- Resolution is always left-to-right in the physical layout
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    ActionAreaType,
    LINE_EXPANSION_SLOTS,
    BUSES_SLOTS,
    PASSENGERS_SLOTS,
    BUILDINGS_SLOTS,
    TIME_CLOCK_SLOTS,
    VRROOMM_SLOTS,
    STARTING_PLAYER_SLOTS,
    ACTION_RESOLUTION_ORDER,
)


# Slot labels in placement order (A first, then B, etc.)
SLOT_LABELS_PLACEMENT_ORDER = ["A", "B", "C", "D", "E", "F"]

# Areas where A is on the right (layout: F-E-D-C-B-A, left to right)
REVERSED_LAYOUT_AREAS = {
    ActionAreaType.LINE_EXPANSION,
    ActionAreaType.BUILDINGS,
}


def get_slot_labels_for_placement(num_slots: int) -> list[str]:
    """Get slot labels in placement order (A first, then B, etc.).

    Args:
        num_slots: Number of slots in the action area.

    Returns:
        List of slot labels in placement order.
    """
    return SLOT_LABELS_PLACEMENT_ORDER[:num_slots]


def get_slot_labels_for_resolution(area_type: ActionAreaType, num_slots: int) -> list[str]:
    """Get slot labels in resolution order (left-to-right physical layout).

    Args:
        area_type: The action area type (determines physical layout).
        num_slots: Number of slots in the action area.

    Returns:
        List of slot labels in left-to-right resolution order.
    """
    labels = SLOT_LABELS_PLACEMENT_ORDER[:num_slots]
    if area_type in REVERSED_LAYOUT_AREAS:
        # F-E-D-C-B-A layout: reverse the labels for resolution
        return list(reversed(labels))
    else:
        # A-B-C-D-E-F layout: labels are already in resolution order
        return labels


@dataclass
class ActionSlot:
    """A single slot in an action area.

    Attributes:
        label: The slot label (A, B, C, D, E, or F).
        player_id: The player who placed a marker here, or None if empty.
        placement_order: The order in which this slot was filled (0-indexed globally).
    """

    label: str
    player_id: Optional[int] = None
    placement_order: Optional[int] = None

    def is_empty(self) -> bool:
        """Check if this slot has no marker."""
        return self.player_id is None

    def place_marker(self, player_id: int, placement_order: int) -> None:
        """Place a marker in this slot.

        Args:
            player_id: The player placing the marker.
            placement_order: Global placement order for this round.

        Raises:
            ValueError: If slot is already occupied.
        """
        if not self.is_empty():
            raise ValueError(f"Slot {self.label} is already occupied by player {self.player_id}")
        self.player_id = player_id
        self.placement_order = placement_order

    def clear(self) -> None:
        """Remove the marker from this slot."""
        self.player_id = None
        self.placement_order = None


@dataclass
class ActionArea:
    """An action area on the action board.

    Attributes:
        area_type: The type of action this area represents.
        slots: Dict mapping slot labels to ActionSlot objects.
        max_slots: Maximum number of slots in this area.
    """

    area_type: ActionAreaType
    slots: dict[str, ActionSlot] = field(default_factory=dict)
    max_slots: int = 1

    def __post_init__(self) -> None:
        """Initialize slots if not provided."""
        if not self.slots:
            labels = get_slot_labels_for_placement(self.max_slots)
            self.slots = {label: ActionSlot(label=label) for label in labels}

    def get_next_available_slot(self) -> Optional[ActionSlot]:
        """Get the next slot available for placement (A first, then B, etc.).

        Returns:
            The next empty slot in placement order, or None if all slots are filled.
        """
        labels = get_slot_labels_for_placement(self.max_slots)
        for label in labels:
            slot = self.slots[label]
            if slot.is_empty():
                return slot
        return None

    def is_full(self) -> bool:
        """Check if all slots are occupied."""
        return all(not slot.is_empty() for slot in self.slots.values())

    def get_slots_in_resolution_order(self) -> list[ActionSlot]:
        """Get all slots in resolution order (left-to-right physical layout)."""
        labels = get_slot_labels_for_resolution(self.area_type, self.max_slots)
        return [self.slots[label] for label in labels]

    def get_occupied_slots_in_resolution_order(self) -> list[ActionSlot]:
        """Get occupied slots in resolution order (left-to-right physical layout)."""
        return [slot for slot in self.get_slots_in_resolution_order() if not slot.is_empty()]

    def get_player_slots(self, player_id: int) -> list[ActionSlot]:
        """Get all slots occupied by a specific player."""
        return [slot for slot in self.slots.values() if slot.player_id == player_id]

    def count_player_markers(self, player_id: int) -> int:
        """Count how many markers a player has in this area."""
        return len(self.get_player_slots(player_id))

    def clear_all(self) -> None:
        """Remove all markers from this area."""
        for slot in self.slots.values():
            slot.clear()


def _get_max_slots(area_type: ActionAreaType) -> int:
    """Get the maximum number of slots for an action area type."""
    return {
        ActionAreaType.LINE_EXPANSION: LINE_EXPANSION_SLOTS,
        ActionAreaType.BUSES: BUSES_SLOTS,
        ActionAreaType.PASSENGERS: PASSENGERS_SLOTS,
        ActionAreaType.BUILDINGS: BUILDINGS_SLOTS,
        ActionAreaType.TIME_CLOCK: TIME_CLOCK_SLOTS,
        ActionAreaType.VRROOMM: VRROOMM_SLOTS,
        ActionAreaType.STARTING_PLAYER: STARTING_PLAYER_SLOTS,
    }[area_type]


@dataclass
class ActionBoard:
    """The complete action board with all action areas.

    Manages marker placement and provides query methods for game logic.

    Attributes:
        areas: Mapping from action area type to ActionArea.
        placement_counter: Tracks global placement order across all areas.
    """

    areas: dict[ActionAreaType, ActionArea] = field(default_factory=dict)
    placement_counter: int = 0

    def __post_init__(self) -> None:
        """Initialize all action areas if not provided."""
        if not self.areas:
            for area_type in ActionAreaType:
                max_slots = _get_max_slots(area_type)
                self.areas[area_type] = ActionArea(
                    area_type=area_type,
                    max_slots=max_slots,
                )

    def get_area(self, area_type: ActionAreaType) -> ActionArea:
        """Get a specific action area."""
        return self.areas[area_type]

    def can_place_marker(self, area_type: ActionAreaType) -> bool:
        """Check if a marker can be placed in the specified area."""
        area = self.get_area(area_type)
        return not area.is_full()

    def place_marker(self, area_type: ActionAreaType, player_id: int) -> ActionSlot:
        """Place a marker in the next available slot of an area.

        Markers are always placed in slot A first, then B, then C, etc.

        Args:
            area_type: The action area to place in.
            player_id: The player placing the marker.

        Returns:
            The slot where the marker was placed.

        Raises:
            ValueError: If the area is full.
        """
        area = self.get_area(area_type)
        slot = area.get_next_available_slot()
        if slot is None:
            raise ValueError(f"Action area {area_type.value} is full")

        slot.place_marker(player_id, self.placement_counter)
        self.placement_counter += 1
        return slot

    def get_available_areas(self) -> list[ActionAreaType]:
        """Get all action areas that have at least one empty slot."""
        return [
            area_type for area_type, area in self.areas.items()
            if not area.is_full()
        ]

    def get_markers_to_resolve(self, area_type: ActionAreaType) -> list[ActionSlot]:
        """Get markers in an area in resolution order (left-to-right physical layout).

        For LINE_EXPANSION and BUILDINGS: F, E, D, C, B, A (left to right)
        For PASSENGERS and VRROOMM: A, B, C, D, E, F (left to right)

        Args:
            area_type: The action area to query.

        Returns:
            List of occupied slots in resolution order.
        """
        area = self.get_area(area_type)
        return area.get_occupied_slots_in_resolution_order()

    def get_all_player_markers(self, player_id: int) -> list[tuple[ActionAreaType, ActionSlot]]:
        """Get all markers placed by a player across all areas.

        Returns:
            List of (area_type, slot) tuples.
        """
        result: list[tuple[ActionAreaType, ActionSlot]] = []
        for area_type, area in self.areas.items():
            for slot in area.get_player_slots(player_id):
                result.append((area_type, slot))
        return result

    def count_total_markers(self) -> int:
        """Count total markers placed on the board."""
        return sum(
            sum(1 for slot in area.slots.values() if not slot.is_empty())
            for area in self.areas.values()
        )

    def clear_all(self) -> None:
        """Remove all markers from all areas (cleanup phase)."""
        for area in self.areas.values():
            area.clear_all()
        self.placement_counter = 0

    def get_resolution_order(self) -> list[ActionAreaType]:
        """Get the fixed order in which action areas are resolved."""
        return ACTION_RESOLUTION_ORDER.copy()
