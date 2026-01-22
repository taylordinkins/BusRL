"""Unit tests for Bus RL wrappers.
"""

import pytest
import numpy as np
from rl.bus_env import BusEnv
from rl.wrappers import BusEnvSelfPlayWrapper


def test_self_play_wrapper_stats():
    """Test that BusEnvSelfPlayWrapper tracks stats correctly."""
    env = BusEnv(num_players=4)
    env = BusEnvSelfPlayWrapper(env)
    
    obs, info = env.reset(seed=42)
    
    # Check initial stats
    for i in range(4):
        assert env.player_stats[i]["reward"] == 0.0
        
    # Take a few steps and check stats accumulation
    terminated = False
    for _ in range(10):
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        
        acting_player = env.unwrapped._engine.state.global_state.current_player_idx
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            break
            
        # Stats should accumulate
        assert env.player_stats[acting_player]["reward"] != 0.0 or reward == 0.0


def test_self_play_wrapper_reset():
    """Test that reset clears per-episode rewards."""
    env = BusEnv(num_players=4)
    env = BusEnvSelfPlayWrapper(env)
    
    env.reset()
    env.player_stats[0]["reward"] = 10.0
    
    env.reset()
    assert env.player_stats[0]["reward"] == 0.0


def test_self_play_wrapper_termination():
    """Test that win stats are updated on termination."""
    env = BusEnv(num_players=4)
    env = BusEnvSelfPlayWrapper(env)
    
    obs, info = env.reset(seed=42)
    
    # Fast-forward to termination (using a seed that might end early or just playing)
    terminated = False
    while not terminated:
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        
    # Winner should be recorded
    assert "winner" in info
    winner = info["winner"]
    assert env.player_stats[winner]["wins"] == 1
    assert "episode_player_rewards" in info
