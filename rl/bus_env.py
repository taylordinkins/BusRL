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
from engine.game_engine import GameEngine, Action, ActionType
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
from .action_space import ActionMapping
from .action_masking import ActionMaskGenerator
from .reward import RewardCalculator


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
        self._action_mapping = ActionMapping(self._board, action_config)
        self._mask_generator = ActionMaskGenerator(self._action_mapping, action_config)
        self._reward_calculator = RewardCalculator(reward_config)

        # State tracking
        self._prev_state: Optional[GameState] = None
        self._current_player_at_step: int = 0
        self._step_count: int = 0
        self._max_steps: int = 2000  # Default max steps per episode
        self._stuck_counter: int = 0
        self._last_state_hash: str = ""

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_config.total_observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(action_config.total_actions)

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

        # Note: RNG seeding for SubprocVecEnv is handled by Gymnasium's seeding
        # mechanism via super().reset(seed=seed). Additional numpy.random.seed()
        # calls can cause all subprocesses to have synchronized RNG states,
        # leading to identical episodes across all environments.

        # Create new game
        self._engine = GameEngine()
        self._engine.reset(num_players=self.num_players, board=self._board.clone())

        # Reset reward calculator
        self._reward_calculator.reset()
        self._step_count = 0
        self._stuck_counter = 0
        self._last_state_hash = self._engine.state.state_hash()

        # FIX 7: Clear cached action mask on reset
        self._cached_action_mask = None

        # Store initial state for reward computation
        self._prev_state = self._engine.state.clone()
        self._current_player_at_step = self._engine.state.global_state.current_player_idx

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Flat action index (0 to total_actions-1).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._engine is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Store state before action for reward computation
        prev_state = self._prev_state
        acting_player = self._engine.state.global_state.current_player_idx

        # Convert action index to Action object
        action_obj = self._action_mapping.index_to_action(action, self._engine.state)

        # 1. Update step count
        self._step_count += 1

        # 2. Execute the action FIRST, then check stuck detection
        if action_obj.params.get("noop", False):
            self._auto_advance()
        else:
            # Execute the action
            step_result = self._engine.step(action_obj)

            if not step_result.success:
                # Invalid action - return penalty
                # Even though action failed, update _prev_state to current state
                # (which should be unchanged from before). This keeps reward calculation
                # consistent and prevents _prev_state from becoming stale.
                self._prev_state = self._engine.state.clone()

                info = self._build_info()
                info["error"] = step_result.info.get("error", "Unknown error")
                info["invalid_action"] = True
                terminated = self._engine.is_game_over()
                return self._get_observation(), self._reward_config.invalid_action_penalty, terminated, False, info

            # Action was successful, now auto-advance through automatic phases
            self._auto_advance()

        # 3. FIX 2: Check stuck detection AFTER action executes
        new_state_hash = self._engine.state.state_hash()
        if new_state_hash == self._last_state_hash:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
            self._last_state_hash = new_state_hash

        # 4. Check termination/truncation after action and auto-advance
        terminated = self._engine.is_game_over()
        # FIX 9: Use > instead of >= for correct max_steps behavior
        truncated = (self._step_count > self._max_steps) or (self._stuck_counter >= 50)

        # Compute reward for the acting player
        action_info = {"action_type": action_obj.action_type.value}
        reward = self._reward_calculator.compute_reward(
            state=self._engine.state,
            prev_state=prev_state,
            player_id=acting_player,
            done=terminated,
            action_info=action_info,
        )

        # Update previous state (this also covers truncation case - FIX 3)
        self._prev_state = self._engine.state.clone()
        self._current_player_at_step = self._engine.state.global_state.current_player_idx

        # Get new observation (from new current player's perspective)
        obs = self._get_observation()

        # FIX 7: Cache action mask to avoid computing multiple times
        self._cached_action_mask = self.action_masks()

        # Build info dict
        info = self._build_info()
        info["acting_player"] = acting_player
        info["reward_breakdown"] = self._reward_calculator.compute_reward_detailed(
            self._engine.state, prev_state, acting_player, terminated, action_info
        ).__dict__

        # If terminal, log extra final info
        if terminated or truncated:
            info["game_over"] = True

        return obs, float(reward), terminated, truncated, info

    def _auto_advance(self) -> None:
        """Auto-advance through phases that don't require player decisions.
    
        This handles phases like CLEANUP and RESOLVING_ACTIONS when
        there are no player choices to make.
        """
        if self._engine is None:
            return

        max_iterations = 100  # Safety limit
        for _ in range(max_iterations):
            if self._engine.is_game_over():
                break

            # Get valid actions for current state
            valid_actions = self._engine.get_valid_actions()

            if len(valid_actions) > 0:
                # Player has choices to make
                break

            # No valid actions - check if we need to advance phases
            phase = self._engine.state.phase

            if phase.value == "cleanup":
                # Execute cleanup and transition
                result = self._engine.resolve_cleanup()
                if not result.success:
                    break
            elif phase.value == "resolving_actions":
                # Auto-resolve all actions using the ActionResolver
                self._auto_resolve_actions()
            else:
                # No actions and not cleanup/resolving - stuck or game over
                break
        
        # Diagnostic: If we hit max iterations, something is very wrong
        # Removed verbose print
        # Removed verbose print
        # Removed verbose print

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

        # FIX 5: Check if resolution succeeded before advancing
        if not result.success:
            # Resolution failed - log but continue (don't crash training)
            # The game may be in an inconsistent state, but the NOOP fallback
            # in action_masks() will handle it
            return

        # Update resolution progress so phase transition triggers
        self._engine.state.global_state.current_resolution_area_idx = len(
            ACTION_RESOLUTION_ORDER
        )

        # Manually trigger phase check to transition to CLEANUP
        self._engine._check_phase_transition()

    def action_masks(self) -> np.ndarray:
        """Get action mask for MaskablePPO.

        Returns:
            Boolean array of shape (total_actions,) where True = valid action.
        """
        if self._engine is None:
            # Return all-invalid mask if not initialized
            return np.zeros(self._action_config.total_actions, dtype=np.bool_)

        valid_actions = self._engine.get_valid_actions()
        mask = self._mask_generator.generate_mask(self._engine.state, valid_actions)

        # Safety check: Ensure at least one action is valid
        # This prevents rare edge cases where all actions are masked
        if not np.any(mask):
            # If truly no valid actions, mark NOOP as valid
            # This should only happen in edge cases during phase transitions
            noop_idx = self._action_config.noop_idx
            mask[noop_idx] = True

        return mask

    def _get_observation(self) -> np.ndarray:
        """Get observation tensor for the current player."""
        if self._engine is None:
            return np.zeros(self._obs_config.total_observation_dim, dtype=np.float32)

        current_player = self._engine.state.global_state.current_player_idx
        return self._obs_encoder.encode(self._engine.state, current_player)

    def _build_info(self) -> dict[str, Any]:
        """Build info dictionary with game metadata using cached mask if available.

        This is the preferred method to call from step() as it uses cached mask.
        """
        if self._engine is None:
            return {}

        state = self._engine.state

        # FIX 7: Use cached mask if available, otherwise compute
        mask = getattr(self, '_cached_action_mask', None)
        if mask is None:
            mask = self.action_masks()

        return {
            "phase": state.phase.value,
            "round": state.global_state.round_number,
            "current_player": state.global_state.current_player_idx,
            "valid_action_count": int(np.sum(mask)),
            "scores": {p.player_id: p.score for p in state.players},
            "time_stones": {p.player_id: p.time_stones for p in state.players},
            "buses": {p.player_id: p.buses for p in state.players},
        }

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary with game metadata.

        Note: For use in step(), prefer _build_info() which uses cached mask.
        """
        return self._build_info()

    def render(self) -> Optional[str]:
        """Render the current state.

        Returns:
            String representation if render_mode is "ansi", None otherwise.
        """
        if self._engine is None:
            return None

        if self.render_mode == "human":
            print(self._engine.state)
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

            # FIX 6: Copy step tracking state for correct truncation behavior
            new_env._step_count = self._step_count
            new_env._stuck_counter = self._stuck_counter
            new_env._last_state_hash = self._last_state_hash

            # Copy cached action mask if present
            new_env._cached_action_mask = (
                self._cached_action_mask.copy()
                if hasattr(self, '_cached_action_mask') and self._cached_action_mask is not None
                else None
            )

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
