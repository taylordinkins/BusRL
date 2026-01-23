"""
Minimal test to diagnose SubprocVecEnv crash.

This script tests the environment creation in isolation to identify
where the EOFError is occurring.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from rl.bus_env import BusEnv
from rl.opponent_pool import OpponentPool, PoolConfig
from rl.multi_policy_env import MultiPolicyBusEnv

def make_env_simple(rank):
    """Simple environment factory without multi-policy."""
    def _init():
        env = BusEnv(num_players=4)
        return env
    return _init

def make_env_multipolicy(rank, pool_dir):
    """Environment factory with multi-policy."""
    def _init():
        print(f"[Subprocess {rank}] Creating environment...")
        env = BusEnv(num_players=4)
        
        print(f"[Subprocess {rank}] Creating opponent pool from {pool_dir}...")
        pool_config = PoolConfig(pool_size=20)
        subprocess_pool = OpponentPool(
            save_dir=pool_dir,
            config=pool_config,
            elo_tracker=None,
        )
        print(f"[Subprocess {rank}] Pool created with {len(subprocess_pool)} checkpoints")
        
        print(f"[Subprocess {rank}] Wrapping with MultiPolicyBusEnv...")
        env = MultiPolicyBusEnv(
            env,
            opponent_pool=subprocess_pool,
            training_slot=0,
            self_play_prob=0.0,
            sampling_method="uniform",
            elo_tracker=None,
            randomize_training_slot=False,
            self_play_checkpoint_path=None,
        )
        print(f"[Subprocess {rank}] Environment created successfully")
        return env
    return _init

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Simple SubprocVecEnv (no multi-policy)")
    print("=" * 60)
    
    try:
        env = SubprocVecEnv([make_env_simple(i) for i in range(2)])
        print("✓ SubprocVecEnv created successfully")
        
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        actions = [env.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = env.step(actions)
        print(f"✓ Step successful")
        
        env.close()
        print("✓ Test 1 PASSED\n")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("Test 2: SubprocVecEnv with MultiPolicyBusEnv")
    print("=" * 60)
    
    # Create a temporary pool directory
    import tempfile
    pool_dir = tempfile.mkdtemp(prefix="test_pool_")
    print(f"Using pool directory: {pool_dir}")
    
    try:
        env = SubprocVecEnv([make_env_multipolicy(i, pool_dir) for i in range(2)])
        print("✓ SubprocVecEnv created successfully")
        
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        actions = [env.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = env.step(actions)
        print(f"✓ Step successful")
        
        env.close()
        print("✓ Test 2 PASSED\n")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        if os.path.exists(pool_dir):
            shutil.rmtree(pool_dir)
    
    print("=" * 60)
    print("All tests complete")
    print("=" * 60)
