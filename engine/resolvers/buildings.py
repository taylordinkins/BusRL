"""Buildings action resolver for the Bus game engine.

This resolver handles the Buildings action area. Players who placed
markers here get to place buildings on the board.

Resolution rules:
- Number of buildings = M#oB - slot_index (A=M#oB, B=M#oB-1, etc.)
- Mandatory inner-first placement: Must place in innermost available zone (A -> B -> C -> D)
- Cannot place in outer zone until all inner zone slots are filled
- Player chooses building type (House, Office, Pub) for each placement
- One building per slot
- Buildings are unowned once placed
- Game ends if all building slots are filled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.constants import ActionAreaType, BuildingType, Zone, ZONE_PRIORITY, TOTAL_BUILDINGS_PER_TYPE
from core.board import NodeId

if TYPE_CHECKING:
    from core.game_state import GameState
    from core.action_board import ActionSlot


# Slot labels in resolution order
SLOT_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@dataclass
class BuildingPlacement:
    """Represents a single building placement.

    Attributes:
        node_id: The node where the building is placed.
        slot_index: The index of the building slot at the node.
        building_type: The type of building to place.
    """

    node_id: NodeId
    slot_index: int
    building_type: BuildingType


@dataclass
class BuildingSlotResult:
    """Result of resolving one slot in the Buildings area.

    Attributes:
        player_id: The player who resolved this slot.
        slot_label: The slot label (A, B, C, etc.).
        buildings_placed: Number of buildings placed.
        placements: List of individual placements made.
    """

    player_id: int
    slot_label: str
    buildings_placed: int
    placements: list[BuildingPlacement]


@dataclass
class BuildingsResult:
    """Result of resolving all markers in the Buildings area.

    Attributes:
        resolved: Whether any markers were resolved.
        slot_results: List of results for each resolved slot.
        total_buildings_placed: Total buildings placed across all slots.
        all_slots_filled: Whether all building slots on the board are now filled.
    """

    resolved: bool
    slot_results: list[BuildingSlotResult] = field(default_factory=list)
    total_buildings_placed: int = 0
    all_slots_filled: bool = False


class BuildingsResolver:
    """Resolves the Buildings action area.

    This resolver:
    1. Gets all markers in the Buildings area in resolution order
    2. For each marker, calculates buildings to place (M#oB - slot_index)
    3. Player places buildings one at a time, choosing type and valid location
    4. Enforces inner-first zone priority (A -> B -> C -> D)

    The number of buildings scales with M#oB and decreases for later slots.
    """

    def __init__(self, state: GameState):
        """Initialize the resolver with the game state.

        Args:
            state: The current game state.
        """
        self.state = state
        self._current_slot_idx = 0
        self._buildings_placed_in_current_slot = 0
        self._slot_results: list[BuildingSlotResult] = []
        self._current_placements: list[BuildingPlacement] = []

    def get_markers_to_resolve(self) -> list[ActionSlot]:
        """Get all markers in the Buildings area in resolution order.

        Returns:
            List of occupied slots in resolution order (left-to-right).
        """
        return self.state.action_board.get_markers_to_resolve(
            ActionAreaType.BUILDINGS
        )

    def has_markers(self) -> bool:
        """Check if there are any markers to resolve in this area."""
        return len(self.get_markers_to_resolve()) > 0

    def get_current_slot(self) -> ActionSlot | None:
        """Get the current slot being resolved.
        
        Automatically skips slots that provide 0 buildings (wasted markers)
        or if all building types have reached their supply limits.

        Returns:
            The current slot, or None if all slots are resolved or no supply.
        """
        # Check if any building type still has supply
        any_supply = False
        for b_type in BuildingType:
            if self.state.board.get_building_count(b_type) < TOTAL_BUILDINGS_PER_TYPE:
                any_supply = True
                break
        
        if not any_supply:
            return None

        markers = self.get_markers_to_resolve()
        
        # Advance past any empty slots
        while self._current_slot_idx < len(markers):
            slot = markers[self._current_slot_idx]
            if self.get_buildings_for_slot(slot) > 0:
                return slot
            self._current_slot_idx += 1
            
        return None

    def get_max_buses(self) -> int:
        """Get the current Maximum Number of Buses (M#oB)."""
        return max(p.buses for p in self.state.players)

    def get_buildings_for_slot(self, slot: ActionSlot) -> int:
        """Calculate the number of buildings for a given slot.

        Formula: M#oB - slot_index (A=0, B=1, etc.)
        Can be 0 if slot_index >= M#oB (wasted marker).
        Also limited by total remaining building supply.

        Args:
            slot: The action slot being resolved.

        Returns:
            Number of buildings to place (0 or more).
        """
        slot_index = SLOT_TO_INDEX.get(slot.label, 0)
        requested = max(0, self.get_max_buses() - slot_index)

        # Calculate total remaining building capacity across all types
        total_remaining = 0
        for b_type in BuildingType:
            count = self.state.board.get_building_count(b_type)
            total_remaining += max(0, TOTAL_BUILDINGS_PER_TYPE - count)
            
        return min(requested, total_remaining)

    def get_buildings_remaining_for_current_slot(self) -> int:
        """Get how many buildings still need to be placed for the current slot."""
        current_slot = self.get_current_slot()
        if current_slot is None:
            return 0
        total = self.get_buildings_for_slot(current_slot)
        return total - self._buildings_placed_in_current_slot

    def get_available_zone(self) -> Zone | None:
        """Get the innermost zone that has available building slots.

        Returns:
            The innermost zone with empty slots, or None if all slots are filled.
        """
        for zone in ZONE_PRIORITY:
            if self.state.board.has_empty_slots_in_zone(zone):
                return zone
        return None

    def get_valid_building_slots(self) -> list[tuple[NodeId, int]]:
        """Get all valid building slots for the current placement.

        Only returns slots in the innermost available zone.

        Returns:
            List of (node_id, slot_index) tuples for valid placements.
        """
        zone = self.get_available_zone()
        if zone is None:
            return []

        valid_slots: list[tuple[NodeId, int]] = []
        for node_id, node in self.state.board.nodes.items():
            for slot_idx, slot in enumerate(node.building_slots):
                if slot.zone == zone and slot.is_empty():
                    valid_slots.append((node_id, slot_idx))

        return valid_slots

    def get_valid_actions(self) -> list[dict]:
        """Get valid actions for Buildings resolution.

        Returns all valid combinations of (node, slot, building_type).

        Returns:
            List of valid action dictionaries.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            return []

        if self.get_buildings_remaining_for_current_slot() <= 0:
            return []

        player_id = current_slot.player_id
        valid_slots = self.get_valid_building_slots()

        actions = []
        for node_id, slot_index in valid_slots:
            for building_type in BuildingType:
                # Only offer building types that still have supply
                if self.state.board.get_building_count(building_type) < TOTAL_BUILDINGS_PER_TYPE:
                    actions.append({
                        "player_id": player_id,
                        "node_id": node_id,
                        "slot_index": slot_index,
                        "building_type": building_type,
                    })

        return actions

    def place_building(self, placement: BuildingPlacement) -> None:
        """Place a single building on the board.

        Args:
            placement: The building placement to execute.

        Raises:
            ValueError: If the placement is invalid.
        """
        current_slot = self.get_current_slot()
        if current_slot is None:
            raise ValueError("No slot to resolve")

        if self.get_buildings_remaining_for_current_slot() <= 0:
            raise ValueError("No buildings remaining for current slot")

        # Validate the placement is in the correct zone
        available_zone = self.get_available_zone()
        if available_zone is None:
            raise ValueError("All building slots are filled")

        node = self.state.board.get_node(placement.node_id)
        if placement.slot_index >= len(node.building_slots):
            raise ValueError(f"Invalid slot index {placement.slot_index} for node {placement.node_id}")

        slot = node.building_slots[placement.slot_index]
        if slot.zone != available_zone:
            raise ValueError(
                f"Must place in zone {available_zone.value}, not zone {slot.zone.value}"
            )

        if not slot.is_empty():
            raise ValueError(f"Slot already contains a building")

        # Validate building type has supply
        if self.state.board.get_building_count(placement.building_type) >= TOTAL_BUILDINGS_PER_TYPE:
            raise ValueError(f"No supply remaining for {placement.building_type.value} buildings")

        # Place the building
        slot.place_building(placement.building_type)

        # Track the placement
        self._current_placements.append(placement)
        self._buildings_placed_in_current_slot += 1

        # Check if we've placed all buildings for this slot
        if self._buildings_placed_in_current_slot >= self.get_buildings_for_slot(current_slot):
            self._finalize_current_slot()

    def _finalize_current_slot(self) -> None:
        """Finalize the current slot and move to the next."""
        current_slot = self.get_current_slot()
        if current_slot is None:
            return

        result = BuildingSlotResult(
            player_id=current_slot.player_id,
            slot_label=current_slot.label,
            buildings_placed=len(self._current_placements),
            placements=list(self._current_placements),
        )
        self._slot_results.append(result)

        # Reset for next slot
        self._current_slot_idx += 1
        self._buildings_placed_in_current_slot = 0
        self._current_placements = []

    def resolve_all(
        self, placements_per_slot: list[list[BuildingPlacement]] | None = None
    ) -> BuildingsResult:
        """Resolve all markers in the Buildings area.

        If placements are not provided, uses default placement
        (House for all buildings in first available slots).

        Args:
            placements_per_slot: Optional list of placements for each slot.

        Returns:
            BuildingsResult with all resolution details.
        """
        markers = self.get_markers_to_resolve()

        if not markers:
            return BuildingsResult(resolved=False)

        # Reset state to ensure clean resolution
        self._current_slot_idx = 0
        self._buildings_placed_in_current_slot = 0
        self._slot_results = []
        self._current_placements = []

        for slot_idx, slot in enumerate(markers):
            # Sync internal slot index with loop index
            self._current_slot_idx = slot_idx
            self._buildings_placed_in_current_slot = 0
            self._current_placements = []

            buildings_to_place = self.get_buildings_for_slot(slot)

            # Skip if no buildings to place
            if buildings_to_place <= 0:
                self._finalize_current_slot()
                continue

            for i in range(buildings_to_place):
                if placements_per_slot and slot_idx < len(placements_per_slot):
                    slot_placements = placements_per_slot[slot_idx]
                    if i < len(slot_placements):
                        self._place_building_internal(slot, slot_placements[i])
                        continue

                # Default: place House in first available slot
                valid_slots = self.get_valid_building_slots()
                if valid_slots:
                    node_id, slot_index = valid_slots[0]
                    self._place_building_internal(slot, BuildingPlacement(
                        node_id=node_id,
                        slot_index=slot_index,
                        building_type=BuildingType.HOUSE,
                    ))
                else:
                    # No valid slots available (all filled)
                    break

            # Always finalize the slot
            self._finalize_current_slot()

        # Update current slot index to reflect completion after loop
        self._current_slot_idx = len(markers)

        total = sum(r.buildings_placed for r in self._slot_results)
        all_filled = self.get_available_zone() is None

        return BuildingsResult(
            resolved=True,
            slot_results=list(self._slot_results),
            total_buildings_placed=total,
            all_slots_filled=all_filled,
        )

    def _place_building_internal(self, slot: "ActionSlot", placement: BuildingPlacement) -> None:
        """Internal method to place building without slot validation.

        Used by resolve_all() which manages slot iteration itself.

        Args:
            slot: The slot being resolved.
            placement: The building placement to execute.
        """
        # Validate the placement is in the correct zone
        available_zone = self.get_available_zone()
        if available_zone is None:
            return  # All slots filled, skip silently

        node = self.state.board.get_node(placement.node_id)
        if placement.slot_index >= len(node.building_slots):
            return  # Invalid slot, skip

        building_slot = node.building_slots[placement.slot_index]
        if building_slot.zone != available_zone:
            return  # Wrong zone, skip

        if not building_slot.is_empty():
            return  # Already occupied, skip

        # Place the building
        building_slot.place_building(placement.building_type)

        # Track the placement
        self._current_placements.append(placement)
        self._buildings_placed_in_current_slot += 1

    def is_resolution_complete(self) -> bool:
        """Check if resolution of this area is complete.

        Returns True when all markers have been resolved.
        """
        markers = self.get_markers_to_resolve()
        if self._current_slot_idx >= len(markers):
            return True

        # Also complete if all slots are filled (game end condition)
        if self.get_available_zone() is None:
            return True

        return False

    def check_game_end_condition(self) -> bool:
        """Check if all building slots are filled (game end condition).

        Returns:
            True if all building slots on the board are filled.
        """
        return self.get_available_zone() is None
