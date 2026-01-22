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
        """Initialize the mask generator.

        Args:
            action_mapping: ActionMapping instance for index conversion.
            config: Action space configuration.
        """
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
                mask[idx] = True
            except ValueError:
                # Action not in our mapping (e.g., resolution actions not yet supported)
                # Skip silently - the environment will handle these separately
                pass

        # FIX 10: Removed redundant NOOP check here.
        # The safety check in BusEnv.action_masks() handles this case
        # unconditionally, which is safer and simpler.

        return mask

    def get_valid_action_indices(self, mask: np.ndarray) -> np.ndarray:
        """Get array of valid action indices from a mask.

        Args:
            mask: Boolean mask array.

        Returns:
            Array of indices where mask is True.
        """
        return np.where(mask)[0]

    def mask_to_logits_mask(
        self,
        mask: np.ndarray,
        invalid_value: float = -1e8,
    ) -> np.ndarray:
        """Convert boolean mask to logits mask for policy networks.

        Args:
            mask: Boolean mask (True = valid).
            invalid_value: Value to use for invalid actions (large negative).

        Returns:
            Float array where valid actions = 0.0, invalid = invalid_value.
        """
        logits_mask = np.where(mask, 0.0, invalid_value)
        return logits_mask.astype(np.float32)

    def count_valid_actions(self, mask: np.ndarray) -> int:
        """Count the number of valid actions in a mask."""
        return int(np.sum(mask))

    def is_action_valid(self, action_idx: int, mask: np.ndarray) -> bool:
        """Check if a specific action index is valid.

        Args:
            action_idx: The action index to check.
            mask: Boolean mask array.

        Returns:
            True if the action is valid.
        """
        if 0 <= action_idx < len(mask):
            return bool(mask[action_idx])
        return False
