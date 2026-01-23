"""Training script for Bus RL agent using MaskablePPO.

This script initializes the Bus environment, sets up a MaskablePPO agent,
and starts the training process with monitoring and checkpoints.

Supports two training modes:
1. Basic self-play: Single policy plays all players (--use-opponent-pool disabled)
2. Multi-policy self-play: Train against diverse opponents from checkpoint pool
   (--use-opponent-pool enabled, optionally with --multi-policy for full PFSP)
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to sys.path to allow importing from rl module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from rl.bus_env import BusEnv
from rl.wrappers import BusEnvSelfPlayWrapper
from rl.opponent_pool import OpponentPool, PoolConfig
from rl.elo_tracker import EloTracker
from rl.multi_policy_env import MultiPolicyBusEnv
from rl.callbacks import (
    OpponentPoolCallback,
    OpponentPoolEvalCallback,
    MultiPolicyTrainingCallback,
)


def make_base_env(num_players: int) -> BusEnv:
    """Create a base BusEnv instance."""
    return BusEnv(num_players=num_players)


class EntropyDecayCallback(BaseCallback):
    """Callback for decaying entropy coefficient over time."""
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Linear decay based on total timesteps
        progress = self.num_timesteps / self.total_timesteps
        ent_coef = self.initial_ent_coef + (self.final_ent_coef - self.initial_ent_coef) * min(progress, 1.0)
        self.model.ent_coef = ent_coef
        if self.num_timesteps % 1000 == 0 and self.logger:
            self.logger.record("train/ent_coef", ent_coef)
        return True


def train(args):
    """Run training for the Bus RL agent."""

    # Create base log directory
    os.makedirs("logs", exist_ok=True)
    run_name = f"ppo_bus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize opponent pool and Elo tracker if enabled
    opponent_pool = None
    elo_tracker = None

    if args.use_opponent_pool:
        pool_config = PoolConfig(
            pool_size=args.pool_size,
            save_interval=args.pool_save_interval,
            prune_strategy=args.prune_strategy,
        )

        # Determine pool directory (use existing if provided, otherwise create new)
        if args.load_pool_dir:
            pool_dir = args.load_pool_dir
            print(f"Loading existing opponent pool from: {pool_dir}")
        else:
            pool_dir = os.path.join(log_dir, "opponent_pool")
            print(f"Creating new opponent pool at: {pool_dir}")

        # Create Elo tracker (will load existing state if present)
        elo_tracker = EloTracker(
            k_factor=args.elo_k_factor,
            initial_elo=1500.0,
            save_path=os.path.join(pool_dir, "elo_state.json"),
        )

        # Create opponent pool (will load existing checkpoints if present)
        opponent_pool = OpponentPool(
            save_dir=pool_dir,
            config=pool_config,
            elo_tracker=elo_tracker,
        )

        print(f"Opponent pool configuration:")
        print(f"  Loaded checkpoints: {len(opponent_pool)}")
        print(f"  Pool size limit: {args.pool_size}")
        print(f"  Save interval: {args.pool_save_interval}")
        print(f"  Prune strategy: {args.prune_strategy}")
        print(f"  Multi-policy mode: {args.multi_policy}")

        if len(opponent_pool) > 0:
            print(f"  Elo range: {opponent_pool.best_elo():.1f} - {opponent_pool.best_elo() - opponent_pool.elo_spread():.1f}")

    # Environment creation function
    def make_env(env_id: str, rank: int, is_eval: bool = False, seed: int = 0):
        """Create environment with unique seeding for each worker process.

        Args:
            env_id: Environment identifier for logging.
            rank: Index of this environment (for unique seeding).
            is_eval: Whether this is an evaluation environment.
            seed: Base seed to use (will be offset by rank).

        Returns:
            Environment factory function.
        """
        def _init():
            env = BusEnv(num_players=args.num_players)

            # For training with multi-policy mode, wrap with MultiPolicyBusEnv
            if args.multi_policy and opponent_pool is not None and not is_eval:
                env = MultiPolicyBusEnv(
                    env,
                    opponent_pool=opponent_pool,
                    training_slot=0,
                    self_play_prob=args.self_play_prob,
                    sampling_method=args.sampling_method,
                    elo_tracker=elo_tracker,
                )
            else:
                env = BusEnvSelfPlayWrapper(env)

            # Monitor should be the outermost wrapper for training stats
            env = Monitor(env)

            # Seed with unique value for this worker.
            # For training envs, use rank-specific seed for diversity.
            # For eval envs, use fixed seed for reproducibility.
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)

            return env
        return _init

    # Wrap for SB3
    # Use a fixed base seed for reproducibility, or generate one
    import time
    base_seed = int(time.time()) % 100000  # Use current time as base seed
    env_factories = [
        make_env(f"train_{i}", rank=i, is_eval=False, seed=base_seed)
        for i in range(args.n_envs)
    ]

    # Use SubprocVecEnv for better performance with multiple envs
    # NOTE: Multi-policy mode requires DummyVecEnv because it needs to share
    # the training_policy object across environments (not possible with multiprocessing)
    if args.multi_policy:
        # FORCE DummyVecEnv for multi-policy to ensure safe object sharing
        # The user reported issues with object propagation in SubprocVecEnv
        print(f"Note: Multi-policy enabled. Forcing DummyVecEnv for correct policy updates.")
        env = DummyVecEnv(env_factories)
        vec_env_class = DummyVecEnv
    elif args.n_envs > 1:
        env = SubprocVecEnv(env_factories)
        vec_env_class = SubprocVecEnv
    else:
        env = DummyVecEnv(env_factories)
        vec_env_class = DummyVecEnv

    # Evaluation environment (uses same vec env type as training for consistency)
    # Use fixed seed for eval to ensure reproducible evaluation
    # Use DummyVecEnv for eval to avoid overhead
    eval_env = DummyVecEnv([make_env("eval", rank=0, is_eval=True, seed=42)])

    # Callbacks
    callbacks = []

    # Standard checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="bus_model",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Entropy decay callback
    if args.ent_coef_final < args.ent_coef:
        entropy_callback = EntropyDecayCallback(
            initial_ent_coef=args.ent_coef,
            final_ent_coef=args.ent_coef_final,
            total_timesteps=args.total_timesteps,
            verbose=1,
        )
        callbacks.append(entropy_callback)

    # Opponent pool callbacks
    if opponent_pool is not None:
        # Checkpoint saving callback
        pool_callback = OpponentPoolCallback(
            opponent_pool=opponent_pool,
            save_interval=args.pool_save_interval,
            verbose=1,
        )
        callbacks.append(pool_callback)

        # Pool evaluation callback (actual head-to-head matches)
        if args.pool_eval_interval > 0:
            pool_eval_callback = OpponentPoolEvalCallback(
                opponent_pool=opponent_pool,
                elo_tracker=elo_tracker,
                env_factory=lambda: make_base_env(args.num_players),
                eval_interval=args.pool_eval_interval,
                n_eval_games=args.pool_eval_games,
                max_opponents=args.pool_eval_opponents,
                verbose=1,
            )
            callbacks.append(pool_eval_callback)

        # Multi-policy training callback
        if args.multi_policy:
            multi_policy_callback = MultiPolicyTrainingCallback(
                opponent_pool=opponent_pool,
                elo_tracker=elo_tracker,
                verbose=1,
            )
            callbacks.append(multi_policy_callback)

    # Initialize model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=min(args.n_steps, args.total_timesteps),
        batch_size=min(args.batch_size, args.total_timesteps),
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        tensorboard_log="logs",
        device=args.device,
    )

    # Set training policy reference for multi-policy environments
    if args.multi_policy and opponent_pool is not None:
        # Access the wrapped environments and set training policy
        # Note: This only works with DummyVecEnv (see env creation logic above)
        if not hasattr(env, 'envs'):
            raise RuntimeError(
                "Multi-policy mode requires DummyVecEnv, but current env doesn't have .envs attribute. "
                "This should not happen - check environment creation logic."
            )

        for i in range(args.n_envs):
            wrapped_env = env.envs[i]
            # Navigate through wrapper stack to find MultiPolicyBusEnv
            current = wrapped_env
            while hasattr(current, 'env'):
                if isinstance(current, MultiPolicyBusEnv):
                    current.training_policy = model
                    break
                current = current.env

    print(f"\nStarting training for {args.total_timesteps} steps...")
    print(f"Logs and models will be saved to: {log_dir}")
    print(f"Number of parallel environments: {args.n_envs}")

    try:
        print("Learning loop starting...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=run_name
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")

    # Save final model
    model.save(os.path.join(log_dir, "bus_model_final"))
    print(f"\nTraining complete. Model saved to {log_dir}")

    # Print opponent pool summary if used
    if opponent_pool is not None:
        print(f"\nOpponent Pool Summary:")
        print(f"  Total checkpoints: {len(opponent_pool)}")
        print(f"  Best Elo: {opponent_pool.best_elo():.1f}")
        print(f"  Elo spread: {opponent_pool.elo_spread():.1f}")

        if elo_tracker is not None:
            leaderboard = elo_tracker.get_leaderboard(top_n=5)
            if leaderboard:
                print(f"\nTop 5 checkpoints by Elo:")
                for checkpoint_id, elo in leaderboard:
                    print(f"  {checkpoint_id}: {elo:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a MaskablePPO agent for Bus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment args
    parser.add_argument("--num-players", type=int, default=4,
                        help="Number of players in game")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")

    # Training args
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total steps to train")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per update")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Initial entropy coefficient")
    parser.add_argument("--ent-coef-final", type=float, default=0.005,
                        help="Final entropy coefficient (if less than initial, decay will occur)")
    parser.add_argument("--target_kl", type=float, default=0.03,
                        help="Target KL divergence")

    # Logging and saving
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Save frequency")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Eval frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Number of eval episodes")
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device")

    # Opponent pool args
    parser.add_argument("--use-opponent-pool", action="store_true",
                        help="Enable opponent pool for self-play diversity")
    parser.add_argument("--load-pool-dir", type=str, default=None,
                        help="Path to existing opponent pool directory to continue from (e.g., logs/ppo_bus_20260122_123456/opponent_pool)")
    parser.add_argument("--pool-size", type=int, default=20,
                        help="Max checkpoints in opponent pool")
    parser.add_argument("--pool-save-interval", type=int, default=50_000,
                        help="Steps between pool checkpoint saves")
    parser.add_argument("--prune-strategy", type=str, default="oldest",
                        choices=["oldest", "lowest_elo", "least_diverse"],
                        help="Strategy for pruning pool when full")

    # Multi-policy training args
    parser.add_argument("--multi-policy", action="store_true",
                        help="Enable multi-policy training (train against pool opponents)")
    parser.add_argument("--self-play-prob", type=float, default=0.2,
                        help="Probability of self-play vs pool opponent")
    parser.add_argument("--sampling-method", type=str, default="pfsp",
                        choices=["uniform", "latest", "pfsp", "elo_weighted"],
                        help="Method for sampling opponents from pool")

    # Pool evaluation args
    parser.add_argument("--pool-eval-interval", type=int, default=100_000,
                        help="Steps between pool evaluation (0 to disable)")
    parser.add_argument("--pool-eval-games", type=int, default=5,
                        help="Games per opponent for pool evaluation")
    parser.add_argument("--pool-eval-opponents", type=int, default=5,
                        help="Max opponents to evaluate against per interval")

    # Elo args
    parser.add_argument("--elo-k-factor", type=float, default=32.0,
                        help="Elo K-factor (higher = more volatile ratings)")

    args = parser.parse_args()

    # Validate args
    if args.multi_policy and not args.use_opponent_pool:
        print("Warning: --multi-policy requires --use-opponent-pool. Enabling opponent pool.")
        args.use_opponent_pool = True

    if args.load_pool_dir:
        # Automatically enable opponent pool if loading from existing directory
        if not args.use_opponent_pool:
            print("Note: --load-pool-dir provided, automatically enabling --use-opponent-pool")
            args.use_opponent_pool = True

        # Validate that the directory exists
        if not os.path.exists(args.load_pool_dir):
            print(f"Error: Pool directory not found: {args.load_pool_dir}")
            sys.exit(1)

        # Check for pool state file
        pool_state_path = os.path.join(args.load_pool_dir, "pool_state.json")
        if not os.path.exists(pool_state_path):
            print(f"Warning: No pool_state.json found in {args.load_pool_dir}")
            print("This may not be a valid opponent pool directory, but continuing anyway...")

    train(args)
