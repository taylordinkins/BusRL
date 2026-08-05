"""Delivery viability utilities for the Bus RL environment.

Provides a reusable, resolver-independent function to compute which passengers
are reachable and deliverable for a given player and building type. Used by
both the observation encoder (when use_delivery_features is enabled) and the
reward calculator (for tiered Vrroomm placement bonuses).

Performance note: get_player_network_nodes() performs a BFS over the player's
edges (O(E+V) at 70 edges). Call compute_delivery_features() once per encoding
pass, not per passenger.

Assumption: We assume the player's network is connected (true in virtually all
post-setup states). A valid delivery source is considered "deliverable" if
available_slot_count > 0 on the network, regardless of exact destination
reachability within the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.constants import BuildingType

if TYPE_CHECKING:
    from core.game_state import GameState


@dataclass
class DeliveryFeatures:
    """Computed delivery viability for one player and one building type."""

    # passenger_id -> True if passenger is on the player's rail network
    passenger_reachable: dict[int, bool]

    # passenger_id -> True if passenger is reachable AND not slot-occupied
    # (i.e., the player can pick them up for delivery right now)
    passenger_valid_source: dict[int, bool]

    # Number of valid sources that have at least one available delivery slot
    deliverable_count: int

    # Number of unoccupied matching building slots on the player's network
    available_slot_count: int


def compute_delivery_features(
    state: "GameState",
    player_id: int,
    building_type: BuildingType,
) -> DeliveryFeatures:
    """Compute delivery viability for a player and building type.

    Called once per (player, building_type) pair. Uses a single BFS to get
    the player's network nodes, then evaluates each passenger and building slot.

    Args:
        state: Current game state.
        player_id: Player whose network to evaluate.
        building_type: The building type to deliver to (current or next clock).

    Returns:
        DeliveryFeatures with per-passenger flags and aggregate counts.
    """
    # Step 1: Get all nodes on the player's rail network (single BFS call)
    network_nodes = state.board.get_player_network_nodes(player_id)

    # Step 2: Simulate slot occupancy — replicate VrrooommResolver logic.
    # Passengers already at a node with a matching building "pre-occupy" that
    # slot, so they are not valid delivery sources.
    occupied_ids: set[int] = set()
    for node_id, node in state.board.nodes.items():
        matching_slots = node.get_buildings_of_type(building_type)
        if not matching_slots or not node.passenger_ids:
            continue
        # Sort passenger IDs ascending (deterministic pairing, mirrors resolver)
        passenger_list = sorted(list(node.passenger_ids))
        for i, passenger_id in enumerate(passenger_list):
            if i < len(matching_slots):
                occupied_ids.add(passenger_id)

    # Step 3: Count unoccupied matching building slots on the player's network.
    # slot.occupied_by_passenger_id is set by VrrooommResolver during resolution
    # and cleared afterwards; during CHOOSING_ACTIONS it is always None.
    available_slot_count = sum(
        1
        for nid in network_nodes
        for slot in state.board.get_node(nid).building_slots
        if slot.building == building_type and slot.occupied_by_passenger_id is None
    )

    # Step 4: Per-passenger reachability and delivery-viability flags.
    passenger_reachable: dict[int, bool] = {}
    passenger_valid_source: dict[int, bool] = {}

    for passenger in state.passenger_manager.passengers.values():
        pid = passenger.passenger_id
        is_reachable = passenger.location in network_nodes
        passenger_reachable[pid] = is_reachable

        if not is_reachable:
            passenger_valid_source[pid] = False
            continue

        # Not in the simulated pre-occupied set
        not_in_occupied = pid not in occupied_ids

        # Not already marked by the resolver as occupying a slot
        # (fallback guard for calls made during Vrroomm resolution itself)
        not_slot_occupied = True
        location_node = state.board.nodes.get(passenger.location)
        if location_node is not None:
            for slot in location_node.building_slots:
                if slot.occupied_by_passenger_id == pid:
                    not_slot_occupied = False
                    break

        passenger_valid_source[pid] = not_in_occupied and not_slot_occupied

    # Step 5: Deliverable count — valid sources that have somewhere to go.
    # Assumes a connected network: any valid source can reach any slot.
    deliverable_count = 0
    if available_slot_count > 0:
        deliverable_count = sum(
            1 for is_valid in passenger_valid_source.values() if is_valid
        )

    return DeliveryFeatures(
        passenger_reachable=passenger_reachable,
        passenger_valid_source=passenger_valid_source,
        deliverable_count=deliverable_count,
        available_slot_count=available_slot_count,
    )
