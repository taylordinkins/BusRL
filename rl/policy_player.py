"""Policy player wrapper for loading trained RL models into the GUI."""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

from core.board import BoardGraph
from core.constants import ActionAreaType
from data.loader import load_default_board
from engine.action_resolver import ActionResolver
from engine.game_engine import GameEngine

from .bus_env import BusEnv
from .config import DEFAULT_OBS_CONFIG, DEFAULT_ACTION_CONFIG, DEFAULT_REWARD_CONFIG
from .hierarchical_action_space import HeadId, VRROOMM_SKIP


# Sentinel dict key used to signal a VRROOMM skip to the GUI.
_VRROOMM_SKIP_KEY = "_vrroomm_skip"


class _EngineResolverBundle:
    """Holds an engine + resolver together so deepcopy preserves their shared state ref."""

    __slots__ = ("engine", "resolver")

    def __init__(self, engine: GameEngine, resolver: ActionResolver) -> None:
        self.engine = engine
        self.resolver = resolver


def is_vrroomm_skip(action: Any) -> bool:
    """Return True if the action dict is a VRROOMM skip sentinel."""
    return isinstance(action, dict) and action.get(_VRROOMM_SKIP_KEY) is True


class PolicyPlayer:
    """Wraps a trained MaskablePPO model for use in the GUI.

    Provides `get_action()` which accepts the GUI's live GameEngine (and
    optional ActionResolver) and returns the action the policy chose:
      - An `Action` object for setup / choosing-actions phases
      - A resolution action dict for resolution phases
      - `{"_vrroomm_skip": True}` when the policy skips VRROOMM deliveries
    """

    def __init__(
        self,
        model: Any,  # MaskablePPO
        num_players: int,
        board: Optional[BoardGraph] = None,
        deterministic: bool = True,
        obs_config=None,
    ) -> None:
        self._model = model
        self._deterministic = deterministic
        resolved_board = board if board is not None else load_default_board()
        resolved_obs_config = obs_config if obs_config is not None else DEFAULT_OBS_CONFIG
        # Private BusEnv used only for obs/mask computation — never stepped.
        self._env = BusEnv(
            num_players=num_players,
            board=resolved_board,
            obs_config=resolved_obs_config,
            action_config=DEFAULT_ACTION_CONFIG,
            reward_config=DEFAULT_REWARD_CONFIG,
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        num_players: int,
        board: Optional[BoardGraph] = None,
        deterministic: bool = True,
    ) -> "PolicyPlayer":
        """Load a MaskablePPO checkpoint and wrap it in a PolicyPlayer."""
        try:
            from sb3_contrib.ppo_mask import MaskablePPO
        except ImportError:
            from sb3_contrib import MaskablePPO  # type: ignore

        import dataclasses
        from .policies import BusMaskableActorCriticPolicy

        model = MaskablePPO.load(
            checkpoint_path,
            device="cpu",
            custom_objects={
                "policy_class": BusMaskableActorCriticPolicy,
                "policy_kwargs": {
                    "logit_clamp": True,
                    "logit_clamp_min": -20.0,
                    "logit_clamp_max": 20.0,
                },
            },
        )

        # Auto-detect which ObservationConfig matches the checkpoint's obs space
        # so that shape mismatches are caught and fixed here rather than at runtime.
        expected_dim = model.observation_space.shape[0]
        obs_config = DEFAULT_OBS_CONFIG
        if obs_config.total_observation_dim != expected_dim:
            alt = dataclasses.replace(obs_config, use_slot_actionability=True)
            if alt.total_observation_dim == expected_dim:
                obs_config = alt
            else:
                raise ValueError(
                    f"Checkpoint obs dim {expected_dim} does not match any known "
                    f"ObservationConfig variant "
                    f"(default={obs_config.total_observation_dim}, "
                    f"with_actionability={alt.total_observation_dim}). "
                    f"The checkpoint may have been trained with a different config."
                )

        return cls(model, num_players, board, deterministic, obs_config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        engine: GameEngine,
        resolver: Optional[ActionResolver] = None,
    ) -> Any:
        """Compute and return the policy's chosen action for the current state.

        Args:
            engine: The GUI's live GameEngine (not mutated — a clone is used internally).
            resolver: The GUI's current ActionResolver.  When provided, it is deep-copied
                      together with the engine in a single pass so that the cloned resolver
                      and engine share the same state object.  This preserves the resolver's
                      current area/slot position rather than restarting from area 0.

        Returns:
            Action object, resolution dict, or VRROOMM-skip sentinel dict.
        """
        # Deep-copy engine and resolver in one pass so deepcopy's memo dict preserves
        # the shared `resolver.state IS engine.state` reference inside the clone.
        if resolver is not None:
            bundle = copy.deepcopy(_EngineResolverBundle(engine, resolver))
            self._env._engine = bundle.engine
            self._env._resolver = bundle.resolver
        else:
            self._env._engine = copy.deepcopy(engine)
            self._env._resolver = None   # will be reconstructed from cloned state
        self._env._decision_cache = None
        self._env._vrroomm_stage_state.complete()  # reset VRROOMM stage

        decision = self._env._get_decision_context()
        if decision is None or decision.get("head_id") is None:
            return None

        head_id: HeadId = decision["head_id"]

        # VRROOMM requires special 2-stage handling.
        if head_id == HeadId.RESOLVE_VRROOMM_PASSENGER:
            return self._handle_vrroomm(decision)

        # All other heads: single predict call.
        obs = self._env._get_observation()
        mask = self._env.action_masks()
        assert mask.any(), "PolicyPlayer: empty action mask — this should never happen"

        action_idx = int(self._model.predict(
            obs, action_masks=mask, deterministic=self._deterministic
        )[0])

        return decision.get("valid_index_to_action", {}).get(action_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_vrroomm(self, passenger_decision: dict) -> Any:
        """Handle the 2-stage VRROOMM delivery selection entirely internally.

        Stage 1: pick a passenger (or skip).
        Stage 2: pick a destination for that passenger.

        Returns a complete delivery dict `{passenger_id, from_node, to_node,
        building_slot_index}` or the VRROOMM-skip sentinel.
        """
        # Stage 1 — select passenger
        obs1 = self._env._get_observation()
        mask1 = self._env.action_masks()
        assert mask1.any(), "PolicyPlayer: empty VRROOMM_PASSENGER mask"

        action_idx1 = int(self._model.predict(
            obs1, action_masks=mask1, deterministic=self._deterministic
        )[0])

        skip_idx = self._env._hier_action_mapping.action_to_index(
            HeadId.RESOLVE_VRROOMM_PASSENGER, VRROOMM_SKIP
        )

        # Skip if policy chose SKIP or no valid passengers.
        valid_passenger_ids: set[int] = passenger_decision.get("valid_passenger_ids", set())
        if action_idx1 == skip_idx or not valid_passenger_ids:
            return {_VRROOMM_SKIP_KEY: True}

        # Decode the passenger id and use tie-break for determinism.
        raw_passenger_id = self._env._hier_action_mapping.index_to_action(
            HeadId.RESOLVE_VRROOMM_PASSENGER, action_idx1
        )
        chosen_passenger_id = self._env._select_vrroomm_passenger_with_tiebreak(
            int(raw_passenger_id), valid_passenger_ids
        )
        # _select_vrroomm_passenger_with_tiebreak advances stage to 2 internally.

        # Stage 2 — select destination
        self._env._decision_cache = None
        decision2 = self._env._get_decision_context()
        if decision2 is None or decision2.get("head_id") != HeadId.RESOLVE_VRROOMM_DEST:
            return {_VRROOMM_SKIP_KEY: True}

        obs2 = self._env._get_observation()
        mask2 = self._env.action_masks()
        if not mask2.any():
            return {_VRROOMM_SKIP_KEY: True}

        action_idx2 = int(self._model.predict(
            obs2, action_masks=mask2, deterministic=self._deterministic
        )[0])

        dest = self._env._hier_action_mapping.index_to_action(
            HeadId.RESOLVE_VRROOMM_DEST, action_idx2
        )
        to_node, building_slot_index = dest

        return self._env._build_vrroomm_delivery_action(int(to_node), int(building_slot_index))
