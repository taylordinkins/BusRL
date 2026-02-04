"""Action masking for the Bus RL environment.

Generates boolean masks indicating which actions are valid in the current
game state. Required for MaskablePPO to ensure only legal actions are sampled.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from engine.game_engine import Action, ActionType
from .config import ActionSpaceConfig, DEFAULT_ACTION_CONFIG

if TYPE_CHECKING:
    from core.game_state import GameState
    from .action_space import ActionMapping


class ActionMaskGenerator:
    """Generates action masks from valid actions.

    The mask is a boolean array of shape (total_actions,) where True
    indicates the action at that index is valid in the current state.
    """

    def __init__(
        self,
        action_mapping: "ActionMapping",
        config: ActionSpaceConfig = DEFAULT_ACTION_CONFIG,
    ):
        self.action_mapping = action_mapping
        self.config = config

    def generate_mask(
        self,
        state: "GameState",
        valid_actions: list[Action],
    ) -> np.ndarray:
        """Generate boolean mask for valid actions.

        Args:
            state: Current game state.
            valid_actions: List of valid Action objects from engine.get_valid_actions().

        Returns:
            Boolean numpy array of shape (total_actions,) where True = valid.
        """
        mask = np.zeros(self.config.total_actions, dtype=np.bool_)

        for action in valid_actions:
            try:
                idx = self.action_mapping.action_to_index(action)
            except ValueError as e:
                # This MUST NOT be allowed to fail silently.
                raise RuntimeError(
                    f"Valid engine action {action} has no ActionMapping entry.\n"
                    "Your action space is inconsistent with the game engine."
                ) from e

            if not (0 <= idx < self.config.total_actions):
                raise RuntimeError(
                    f"ActionMapping produced out-of-bounds index {idx} for action {action}"
                )

            mask[idx] = True

        # Fallback for forced auto-advance only
        if not mask.any():
            if state.is_game_over():
                # Terminal state → allow NOOP so PPO has a legal action
                mask[self.config.noop_idx] = True
            else:
                raise RuntimeError(
                    "No valid actions produced for a non-terminal game state.\n"
                    "This indicates a game engine or masking bug."
                )

        assert mask.any()
        return mask

    def get_valid_action_indices(self, mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0]

    def mask_to_logits_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert boolean mask to logits mask.

        Valid actions → 0.0
        Invalid actions → -1e8 (matches sb3-contrib MaskableCategorical)
        """
        logits_mask = np.full(mask.shape, -1e8, dtype=np.float32)
        logits_mask[mask] = 0.0
        return logits_mask

    def count_valid_actions(self, mask: np.ndarray) -> int:
        return int(mask.sum())

    def is_action_valid(self, action_idx: int, mask: np.ndarray) -> bool:
        return 0 <= action_idx < len(mask) and bool(mask[action_idx])
