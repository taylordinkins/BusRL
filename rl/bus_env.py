"""Gymnasium environment for the Bus board game.

Provides a single-agent, turn-based interface for RL training with:
- Shared policy across all players (self-play)
- Observations from current player's perspective
- Action masking for legal move enforcement
- Compatible with MaskablePPO from sb3-contrib
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, SupportsFloat
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fall back to gym if gymnasium not available
    import gym
    from gym import spaces

from core.game_state import GameState
from core.board import BoardGraph
from core.constants import (
    Phase,
    ActionAreaType,
    ACTION_RESOLUTION_ORDER,
    MIN_MARKERS_PER_ROUND,
)
from engine.game_engine import GameEngine, Action, ActionType
from engine.action_resolver import ActionResolver, ResolutionStatus
from data.loader import load_default_board

from .config import (
    ObservationConfig,
    ActionSpaceConfig,
    RewardConfig,
    DEFAULT_OBS_CONFIG,
    DEFAULT_ACTION_CONFIG,
    DEFAULT_REWARD_CONFIG,
)
from .observation import ObservationEncoder
from .hierarchical_action_space import (
    HierarchicalActionMapping,
    HeadId,
    get_head_id,
    VRROOMM_SKIP,
)
from .reward import RewardCalculator
from .vrroomm_stage import VrroommStageState


class BusEnv(gym.Env):
    """Gymnasium environment for the Bus board game.

    This environment implements a single-agent, turn-based interface where:
    - All players share the same policy (self-play)
    - Observations are from the current player's perspective
    - Action masking ensures only legal actions are sampled
    - The environment auto-advances through phases with no player choices

    Compatible with stable-baselines3's MaskablePPO via the action_masks() method.

    Attributes:
        observation_space: Box space for flat observation tensor.
        action_space: Discrete space for all possible actions.
        metadata: Environment metadata including render modes.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_players: int = 4,
        board: Optional[BoardGraph] = None,
        render_mode: Optional[str] = None,
        obs_config: ObservationConfig = DEFAULT_OBS_CONFIG,
        action_config: ActionSpaceConfig = DEFAULT_ACTION_CONFIG,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ):
        """Initialize the Bus environment.

        Args:
            num_players: Number of players (3-5).
            board: Optional custom board. Uses default if None.
            render_mode: Rendering mode ("human", "ansi", or None).
            obs_config: Observation encoding configuration.
            action_config: Action space configuration.
            reward_config: Reward calculation configuration.
        """
        super().__init__()

        self.num_players = num_players
        self._board = board if board is not None else load_default_board()
        self.render_mode = render_mode

        # Configuration
        self._obs_config = obs_config
        self._action_config = action_config
        self._reward_config = reward_config

        # Core components
        self._engine: Optional[GameEngine] = None
        self._obs_encoder = ObservationEncoder(obs_config)
        self._hier_action_mapping = HierarchicalActionMapping(self._board)
        self._max_head_actions = max(
            self._hier_action_mapping.head_size(head_id) for head_id in HeadId
        )
        self._reward_calculator = RewardCalculator(reward_config)
        self._resolver: Optional[ActionResolver] = None
        self._decision_cache: Optional[dict[str, Any]] = None
        self._vrroomm_stage_state = VrroommStageState()

        # State tracking
        self._prev_state: Optional[GameState] = None
        self._current_player_at_step: int = 0
        self._step_count: int = 0
        self._max_steps: int = 2000  # Default max steps per episode
        self._active_step_count: int = 0
        self._stuck_counter: int = 0
        self._last_state_hash: str = ""
        # For logging true episode lengths
        self._episode_lengths: list[int] = []
        # Resolution waste tracking (per round)
        self._resolution_waste_round: Optional[int] = None
        self._resolution_waste_by_area: dict[str, dict[str, int]] = {}
        self._resolution_waste_total: int = 0

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_config.total_observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._max_head_actions)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Create new game
        self._engine = GameEngine()
        self._engine.reset(num_players=self.num_players, board=self._board.clone())

        # Reset reward calculator
        self._reward_calculator.reset()
        self._resolver = None
        self._decision_cache = None
        self._vrroomm_stage_state.complete()
        self._resolution_waste_round = None
        self._resolution_waste_by_area = {}
        self._resolution_waste_total = 0

        # Episode bookkeeping
        self._step_count = 0

        # Store initial state for reward computation
        self._prev_state = self._engine.state.clone()
        self._current_player_at_step = self._engine.state.global_state.current_player_idx

        # Initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info


    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Execute one step in the environment with hierarchical action flow."""
        if self._engine is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        prev_state = self._prev_state
        acting_player = self._engine.state.global_state.current_player_idx
        info = self._get_info()

        # Increment step count
        self._step_count += 1

        decision = self._get_decision_context()
        head_id = decision.get("head_id") if decision else None
        mask = decision.get("mask") if decision else None

        if isinstance(head_id, HeadId):
            info["action_head_id"] = head_id.value
            info["action_head_name"] = head_id.name
        else:
            info["action_head_id"] = None
            info["action_head_name"] = None

        if head_id is None or mask is None:
            # No decision required; auto-advance and return
            self._advance_after_action()
            terminated = self._engine.is_game_over()
            truncated = self._step_count >= self._max_steps
            reward = 0.0
            obs = self._get_observation()
            return obs, reward, terminated, truncated, info

        # Validate chosen action against mask
        if not (0 <= action < len(mask)) or not mask[action]:
            info["invalid_action"] = True
            self._prev_state = self._engine.state.clone()
            obs = self._get_observation()
            return (
                obs,
                self._reward_config.invalid_action_penalty,
                False,
                False,
                info,
            )

        action_info: dict[str, Any] = {"action_type": head_id.name}

        # Apply action based on head
        if head_id in (
            HeadId.SETUP_BUILDINGS,
            HeadId.SETUP_RAILS_FORWARD,
            HeadId.SETUP_RAILS_REVERSE,
            HeadId.CHOOSING_ACTIONS,
        ):
            if action not in decision["valid_index_to_action"]:
                raise RuntimeError("Chosen action not in valid action mapping")
            action_obj = decision["valid_index_to_action"][action]
            if head_id == HeadId.CHOOSING_ACTIONS and action_obj.action_type == ActionType.PASS:
                current_player = self._engine.state.get_current_player()
                if not (
                    current_player.markers_placed_this_round >= MIN_MARKERS_PER_ROUND
                    or current_player.action_markers_remaining == 0
                ):
                    raise RuntimeError("PASS chosen before minimum marker placement")
            if head_id == HeadId.CHOOSING_ACTIONS and action_obj.action_type == ActionType.PLACE_MARKER:
                area_type = ActionAreaType(action_obj.params["area_type"])
                if self._engine is not None:
                    next_slot = self._engine.state.action_board.get_area(area_type).get_next_available_slot()
                    if next_slot is not None:
                        info["placed_marker_area"] = area_type.value
                        info["placed_marker_slot"] = next_slot.label
            step_result = self._engine.step(action_obj)
            if not step_result.success:
                info["error"] = step_result.info.get("error", "Unknown error")
                info["invalid_action"] = True
                self._prev_state = self._engine.state.clone()
                obs = self._get_observation()
                return (
                    obs,
                    self._reward_config.invalid_action_penalty,
                    False,
                    False,
                    info,
                )

        elif head_id == HeadId.RESOLVE_VRROOMM_PASSENGER:
            skip_idx = self._hier_action_mapping.action_to_index(
                HeadId.RESOLVE_VRROOMM_PASSENGER, VRROOMM_SKIP
            )
            if action == skip_idx:
                if self._resolver is not None:
                    self._resolver.skip_vrroomm_deliveries()
                self._skip_vrroomm()
            else:
                valid_passenger_ids = decision["valid_passenger_ids"]
                passenger_id = self._hier_action_mapping.index_to_action(
                    HeadId.RESOLVE_VRROOMM_PASSENGER, action
                )
                if passenger_id not in valid_passenger_ids:
                    raise RuntimeError("Chosen passenger not in valid resolver actions")
                self._select_vrroomm_passenger_with_tiebreak(
                    int(passenger_id), valid_passenger_ids
                )

        elif head_id == HeadId.RESOLVE_VRROOMM_DEST:
            valid_destinations = decision["valid_destinations"]
            dest = self._hier_action_mapping.index_to_action(
                HeadId.RESOLVE_VRROOMM_DEST, action
            )
            if dest not in valid_destinations:
                raise RuntimeError("Chosen destination not in valid resolver actions")

            if self._resolver is None:
                raise RuntimeError("Resolver missing during Vrroomm destination selection")

            to_node, slot_idx = dest
            delivery_action = self._build_vrroomm_delivery_action(to_node, slot_idx)
            action_info["delivery"] = delivery_action
            self._resolver.apply_action(delivery_action)

        else:
            if self._resolver is None:
                raise RuntimeError("Resolver missing during resolution action")
            if action not in decision["valid_index_to_action"]:
                raise RuntimeError("Chosen action not in valid action mapping")
            action_dict = decision["valid_index_to_action"][action]
            self._resolver.apply_action(action_dict)

        # Clear decision cache after applying action
        self._decision_cache = None

        # Advance automatic phases
        self._advance_after_action()

        # Re-check terminal and truncation
        terminated = self._engine.is_game_over()
        truncated = self._step_count >= self._max_steps
        self._active_step_count += 1

        # Compute reward
        reward = self._reward_calculator.compute_reward(
            state=self._engine.state,
            prev_state=prev_state,
            player_id=acting_player,
            done=terminated,
            action_info=action_info,
        )

        # Update previous state
        self._prev_state = self._engine.state.clone()
        self._current_player_at_step = self._engine.state.global_state.current_player_idx

        # Observation
        obs = self._get_observation()

        # Merge info
        new_info = self._get_info()
        info.update(new_info)
        info["next_head_id"] = new_info.get("head_id")
        info["next_head_name"] = new_info.get("head_name")
        info["head_id"] = info.get("action_head_id")
        info["head_name"] = info.get("action_head_name")
        info["acting_player"] = acting_player
        info["reward_breakdown"] = self._reward_calculator.compute_reward_detailed(
            self._engine.state, prev_state, acting_player, terminated, action_info
        ).__dict__

        if terminated or truncated:
            self._episode_lengths.append(self._active_step_count)
            self._active_step_count = 0
            info["game_over"] = True

        return obs, float(reward), terminated, truncated, info


    def _auto_advance(self) -> None:
        """Auto-advance through phases that do not require player decisions."""
        if self._engine is None:
            return
        if self._engine.state.phase == Phase.CLEANUP:
            self._engine.resolve_cleanup()


    def _auto_resolve_actions(self) -> None:
        """Auto-resolve the RESOLVING_ACTIONS phase.

        Uses the ActionResolver to step through all resolution areas.
        For RL training, we use default choices for player decisions
        (e.g., time clock advances, first valid rail placement, etc.)
        """
        from engine.action_resolver import ActionResolver
        from core.constants import ACTION_RESOLUTION_ORDER

        resolver = ActionResolver(self._engine.state)
        result = resolver.resolve_all()

        # Update resolution progress so phase transition triggers
        self._engine.state.global_state.current_resolution_area_idx = len(
            ACTION_RESOLUTION_ORDER
        )

        # Manually trigger phase check to transition to CLEANUP
        self._engine._check_phase_transition()

    # -------------------------------------------------------------------------
    # Hierarchical resolution helpers
    # -------------------------------------------------------------------------

    def _sync_resolution_state(self) -> None:
        """Sync resolver context into GlobalState for phase transitions."""
        if self._engine is None or self._resolver is None:
            return
        ctx = self._resolver.get_context()
        if ctx.current_area is None:
            # Resolution complete
            self._engine.state.global_state.current_resolution_area_idx = len(
                ACTION_RESOLUTION_ORDER
            )
            self._engine.state.global_state.current_resolution_slot_idx = 0
            return

        self._engine.state.global_state.current_resolution_area_idx = ctx.current_area_idx
        self._engine.state.global_state.current_resolution_slot_idx = ctx.current_slot_idx
        if ctx.awaiting_player_id is not None:
            self._engine.state.global_state.current_player_idx = ctx.awaiting_player_id

    def _ensure_resolver(self) -> Optional[ActionResolver]:
        if self._engine is None:
            return None
        if self._engine.state.phase != Phase.RESOLVING_ACTIONS:
            self._resolver = None
            self._vrroomm_stage_state.complete()
            return None
        if self._resolver is None:
            self._resolver = ActionResolver(self._engine.state)
            self._resolver.start_resolution()
            self._sync_resolution_state()
        return self._resolver

    def _advance_resolver_until_input(self) -> Optional["ResolutionContext"]:
        resolver = self._ensure_resolver()
        if resolver is None:
            return None

        while True:
            ctx = resolver.get_context()
            self._sync_resolution_state()
            self._maybe_record_resolution_waste(ctx)

            if ctx.status == ResolutionStatus.AWAITING_INPUT:
                if ctx.current_area == ActionAreaType.VRROOMM:
                    if self._vrroomm_stage_state.stage == 0:
                        self._enter_vrroomm()
                else:
                    self._vrroomm_stage_state.complete()
                return ctx

            if ctx.status == ResolutionStatus.ALL_COMPLETE:
                self._vrroomm_stage_state.complete()
                # Ensure phase transition to CLEANUP
                self._engine._check_phase_transition()
                return ctx

            resolver.advance()

    def _maybe_record_resolution_waste(self, ctx: "ResolutionContext") -> None:
        if self._engine is None or ctx.current_area is None:
            return
        if self._resolution_waste_round != self._engine.state.global_state.round_number:
            self._resolution_waste_round = self._engine.state.global_state.round_number
            self._resolution_waste_by_area = {}
            self._resolution_waste_total = 0

        area = ctx.current_area
        if area not in (
            ActionAreaType.LINE_EXPANSION,
            ActionAreaType.PASSENGERS,
            ActionAreaType.BUILDINGS,
        ):
            return

        area_key = area.value
        if area_key in self._resolution_waste_by_area:
            return

        max_buses = max(p.buses for p in self._engine.state.players)
        area_obj = self._engine.state.action_board.get_area(area)

        slot_index_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        total_markers = 0
        wasted_markers = 0

        for slot in area_obj.slots.values():
            if slot.player_id is None:
                continue
            total_markers += 1
            slot_index = slot_index_map.get(slot.label, 0)
            if (max_buses - slot_index) <= 0:
                wasted_markers += 1

        self._resolution_waste_by_area[area_key] = {
            "total": total_markers,
            "wasted": wasted_markers,
            "max_buses": max_buses,
        }
        self._resolution_waste_total = sum(
            item["wasted"] for item in self._resolution_waste_by_area.values()
        )

    def _advance_after_action(self) -> None:
        if self._engine is None:
            return
        if self._engine.state.phase == Phase.RESOLVING_ACTIONS:
            self._advance_resolver_until_input()
        if self._engine.state.phase == Phase.CLEANUP:
            self._engine.resolve_cleanup()

    def _build_mask(
        self, head_id: HeadId, valid_actions: list[Any]
    ) -> tuple[np.ndarray, dict[int, Any]]:
        """Build a mask and index mapping from valid actions for a head."""
        mask = np.zeros(self._max_head_actions, dtype=np.bool_)
        valid_index_to_action: dict[int, Any] = {}
        for action in valid_actions:
            idx = self._hier_action_mapping.action_to_index(head_id, action)
            mask[idx] = True
            valid_index_to_action[idx] = action
        return mask, valid_index_to_action

    def _get_decision_context(self, _retry: bool = False) -> Optional[dict[str, Any]]:
        """Compute and cache current decision context."""
        if self._decision_cache is not None:
            return self._decision_cache

        if self._engine is None:
            return None

        phase = self._engine.state.phase
        resolver_ctx = None
        resolution_area = None
        valid_actions: list[Any] = []

        if phase == Phase.RESOLVING_ACTIONS:
            resolver_ctx = self._advance_resolver_until_input()
            if resolver_ctx is None:
                return None
            resolution_area = resolver_ctx.current_area
            if resolver_ctx.status == ResolutionStatus.ALL_COMPLETE:
                self._decision_cache = {"head_id": None, "mask": None}
                return self._decision_cache
            valid_actions = list(resolver_ctx.valid_actions)
        else:
            valid_actions = list(self._engine.get_valid_actions())

        head_id = get_head_id(
            phase, resolution_area, self._vrroomm_stage_state.stage
        )
        if head_id is None:
            self._decision_cache = {"head_id": None, "mask": None}
            return self._decision_cache

        mask = np.zeros(self._max_head_actions, dtype=np.bool_)
        valid_index_to_action: dict[int, Any] = {}
        valid_passenger_ids: set[int] = set()
        valid_destinations: set[tuple[int, int]] = set()

        if head_id == HeadId.RESOLVE_VRROOMM_PASSENGER:
            valid_passenger_ids = {
                int(a["passenger_id"]) for a in valid_actions if "passenger_id" in a
            }
            for passenger_id in valid_passenger_ids:
                idx = self._hier_action_mapping.action_to_index(
                    HeadId.RESOLVE_VRROOMM_PASSENGER, passenger_id
                )
                mask[idx] = True
            skip_idx = self._hier_action_mapping.action_to_index(
                HeadId.RESOLVE_VRROOMM_PASSENGER, VRROOMM_SKIP
            )
            mask[skip_idx] = True

        elif head_id == HeadId.RESOLVE_VRROOMM_DEST:
            selected_id = self._vrroomm_stage_state.selected_passenger_id
            if selected_id is not None:
                valid_destinations = {
                    (int(a["to_node"]), int(a["building_slot_index"]))
                    for a in valid_actions
                    if int(a["passenger_id"]) == selected_id
                }
            for dest in valid_destinations:
                idx = self._hier_action_mapping.action_to_index(
                    HeadId.RESOLVE_VRROOMM_DEST, dest
                )
                mask[idx] = True

        else:
            mask, valid_index_to_action = self._build_mask(head_id, valid_actions)

        if mask.sum() > self._hier_action_mapping.head_size(head_id):
            raise RuntimeError("Mask leakage: mask.sum() exceeds head size")
        if not mask.any():
            if phase == Phase.RESOLVING_ACTIONS and not _retry:
                self._decision_cache = None
                self._advance_resolver_until_input()
                return self._get_decision_context(_retry=True)
            raise RuntimeError("Empty action mask in decision phase")

        self._decision_cache = {
            "head_id": head_id,
            "mask": mask,
            "valid_actions": valid_actions,
            "valid_index_to_action": valid_index_to_action,
            "valid_passenger_ids": valid_passenger_ids,
            "valid_destinations": valid_destinations,
        }
        return self._decision_cache

    def action_masks(self) -> np.ndarray:
        """Get action mask for hierarchical PPO.

        Returns:
            Boolean array of shape (max_head_actions,) where True = valid action.
        """
        if self._engine is None:
            mask = np.zeros(self._max_head_actions, dtype=np.bool_)
            mask[0] = True
            return mask

        decision = self._get_decision_context()
        if decision is None or decision.get("mask") is None:
            mask = np.zeros(self._max_head_actions, dtype=np.bool_)
            mask[0] = True
            return mask

        action_mask = decision["mask"]
        if np.any(np.isnan(action_mask)):
            raise RuntimeError("NaN IN MASK")
        if not np.any(action_mask):
            raise RuntimeError("EMPTY ACTION MASK — THIS WILL CRASH PPO")

        return action_mask


    def _get_observation(self) -> np.ndarray:
        """Get observation tensor for the current player."""
        if self._engine is None:
            return np.zeros(self._obs_config.total_observation_dim, dtype=np.float32)

        current_player = self._engine.state.global_state.current_player_idx
        decision = self._get_decision_context()
        head_id = decision.get("head_id") if decision else None
        return self._obs_encoder.encode(
            self._engine.state,
            current_player,
            head_id=head_id,
            vrroomm_stage=self._vrroomm_stage_state.stage,
        )

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary with game metadata."""
        if self._engine is None:
            return {}

        state = self._engine.state
        decision = self._get_decision_context()
        mask = decision["mask"] if decision and decision.get("mask") is not None else None
        valid_action_count = int(np.sum(mask)) if mask is not None else 0
        head_id = decision.get("head_id") if decision else None

        return {
            "phase": state.phase.value,
            "round": state.global_state.round_number,
            "current_player": state.global_state.current_player_idx,
            "valid_action_count": valid_action_count,
            "head_id": head_id.value if isinstance(head_id, HeadId) else None,
            "head_name": head_id.name if isinstance(head_id, HeadId) else None,
            "vrroomm_stage": self._vrroomm_stage_state.stage,
            "resolution_waste_by_area": self._resolution_waste_by_area or None,
            "resolution_waste_total": self._resolution_waste_total,
            "scores": {p.player_id: p.score for p in state.players},
            "time_stones": {p.player_id: p.time_stones for p in state.players},
            "buses": {p.player_id: p.buses for p in state.players},
        }

    def render(self) -> Optional[str]:
        """Render the current state.

        Returns:
            String representation if render_mode is "ansi", None otherwise.
        """
        if self._engine is None:
            return None

        if self.render_mode == "human":
            #print(self._engine.state)
            return None
        elif self.render_mode == "ansi":
            return str(self._engine.state)
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        self._engine = None
        self._prev_state = None

    # -------------------------------------------------------------------------
    # Additional utility methods
    # -------------------------------------------------------------------------

    def get_state(self) -> Optional[GameState]:
        """Get the current game state (for debugging/visualization)."""
        if self._engine is None:
            return None
        return self._engine.state

    def get_valid_actions(self) -> list[Action]:
        """Get list of valid Action objects (for debugging)."""
        if self._engine is None:
            return []
        return self._engine.get_valid_actions()

    def get_current_player(self) -> int:
        """Get the current player ID."""
        if self._engine is None:
            return 0
        return self._engine.state.global_state.current_player_idx

    # -------------------------------------------------------------------------
    # Vrroomm two-stage helpers (hierarchical action flow)
    # -------------------------------------------------------------------------

    def get_vrroomm_stage(self) -> int:
        """Get current Vrroomm stage (0=none, 1=passenger, 2=destination)."""
        return self._vrroomm_stage_state.stage

    def _enter_vrroomm(self) -> None:
        self._vrroomm_stage_state.enter()

    def _select_vrroomm_passenger(self, passenger_id: int) -> None:
        self._vrroomm_stage_state.select_passenger(passenger_id)

    def _select_vrroomm_passenger_with_tiebreak(
        self, passenger_id: int, valid_passenger_ids: Optional[set[int]] = None
    ) -> int:
        """Select a passenger with origin-based tie-break.

        If multiple passengers are at the same origin, choose the lowest
        passenger_id at that origin to keep selection deterministic.
        """
        if self._engine is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        passenger = self._engine.state.passenger_manager.get_passenger(int(passenger_id))
        if passenger is None:
            raise ValueError(f"Passenger {passenger_id} not found")

        origin = passenger.location
        passengers_at_origin = self._engine.state.passenger_manager.get_passengers_at(origin)
        if not passengers_at_origin:
            raise ValueError(f"No passengers at origin {origin}")

        if valid_passenger_ids is not None and passenger_id not in valid_passenger_ids:
            raise ValueError("Passenger not in resolver-valid list")

        candidate_ids = [p.passenger_id for p in passengers_at_origin]
        if valid_passenger_ids is not None:
            candidate_ids = [pid for pid in candidate_ids if pid in valid_passenger_ids]

        if not candidate_ids:
            raise ValueError("No resolver-valid passengers at origin")

        chosen_id = min(candidate_ids)
        self._select_vrroomm_passenger(chosen_id)
        return chosen_id

    def _build_vrroomm_delivery_action(
        self,
        to_node: int,
        building_slot_index: int,
    ) -> dict:
        if self._engine is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._vrroomm_stage_state.build_delivery_action(
            self._engine.state, to_node, building_slot_index
        )

    def _complete_vrroomm(self) -> None:
        self._vrroomm_stage_state.complete()

    def _skip_vrroomm(self) -> None:
        self._vrroomm_stage_state.skip()

    def clone(self) -> "BusEnv":
        """Create a deep copy of the environment.

        Useful for MCTS or other search algorithms.
        """
        new_env = BusEnv(
            num_players=self.num_players,
            board=self._board,
            render_mode=self.render_mode,
            obs_config=self._obs_config,
            action_config=self._action_config,
            reward_config=self._reward_config,
        )

        if self._engine is not None:
            new_env._engine = self._engine.clone()
            new_env._prev_state = self._prev_state.clone() if self._prev_state else None
            new_env._current_player_at_step = self._current_player_at_step
            # Copy reward calculator state
            new_env._reward_calculator._stations_connected = {
                k: v.copy()
                for k, v in self._reward_calculator._stations_connected.items()
            }

        return new_env


def make_bus_env(
    num_players: int = 4,
    render_mode: Optional[str] = None,
    **kwargs,
) -> BusEnv:
    """Factory function for creating Bus environments.

    Args:
        num_players: Number of players (3-5).
        render_mode: Rendering mode.
        **kwargs: Additional arguments passed to BusEnv.

    Returns:
        Configured BusEnv instance.
    """
    return BusEnv(num_players=num_players, render_mode=render_mode, **kwargs)
