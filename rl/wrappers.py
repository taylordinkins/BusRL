"""Wrappers for the Bus RL environment.

This module provides Gymnasium wrappers for self-play, observation handling,
and stable training utilities.
"""

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class BusEnvSelfPlayWrapper(gym.Wrapper):
    """Wrapper for self-play training on BusEnv.
    
    In a single-agent turn-based environment like BusEnv, every step is
    taken by the "current player". This wrapper ensures that researchers
    can track performance across all player slots independently.
    
    It also provides a hook for future multi-agent integration where
    different policies might control different players.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.player_stats = {i: {"reward": 0.0, "wins": 0} for i in range(5)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        for i in range(5):
            self.player_stats[i]["reward"] = 0.0
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # The env.step() returns reward for the player who just acted
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        acting_player = info.get("acting_player", 0)
        self.player_stats[acting_player]["reward"] += reward
        
        if terminated:
            # Update win stats based on scores
            scores = info.get("scores", {})
            if scores:
                winner = max(scores, key=scores.get)
                self.player_stats[winner]["wins"] += 1
                info["winner"] = winner
            
            # Add episode cumulative rewards to info for logging
            info["episode_player_rewards"] = {
                i: self.player_stats[i]["reward"] for i in range(self.env.unwrapped.num_players)
            }
            
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Forward action masks from the underlying environment."""
        try:
            return self.env.action_masks()
        except AttributeError:
            return self.env.get_wrapper_attr("action_masks")()


class RewardNormalizer(gym.RewardWrapper):
    """Normalizes rewards by a constant factor to stabilize training.
    
    Useful if delivery rewards (+1.0) and point differentials (+X) 
    have very different scales.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale

    def action_masks(self) -> np.ndarray:
        """Forward action masks from the underlying environment."""
        try:
            return self.env.action_masks()
        except AttributeError:
            return self.env.get_wrapper_attr("action_masks")()
