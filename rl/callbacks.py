"""Custom callbacks for RL training with opponent pool integration.

This module provides callbacks that integrate the opponent pool with
the training loop, enabling automatic checkpoint saving, pool management,
and evaluation against pool opponents with Elo rating updates.
"""

from __future__ import annotations

import os
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .opponent_pool import OpponentPool
    from .elo_tracker import EloTracker


class OpponentPoolCallback(BaseCallback):
    """Callback for managing the opponent pool during training.

    This callback:
    - Saves checkpoints to the opponent pool at regular intervals
    - Logs pool statistics to TensorBoard
    - Can be extended for PFSP matchmaking in the future

    Example:
        >>> pool = OpponentPool(save_dir="checkpoints")
        >>> callback = OpponentPoolCallback(pool, save_interval=50000)
        >>> model.learn(total_timesteps=1000000, callback=callback)
    """

    def __init__(
        self,
        opponent_pool: "OpponentPool",
        save_interval: int = 50_000,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: The opponent pool to manage.
            save_interval: Training steps between checkpoint saves.
            verbose: Verbosity level (0 = no output, 1 = info).
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.save_interval = save_interval
        self._last_save_step = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if it's time to save a checkpoint
        current_step = self.num_timesteps

        if current_step - self._last_save_step >= self.save_interval:
            self._save_checkpoint()
            self._last_save_step = current_step

        return True

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        # Set the current policy in the pool
        self.opponent_pool.current_policy = self.model

        # Save initial checkpoint
        if len(self.opponent_pool) == 0:
            self._save_checkpoint(is_initial=True)

        if self.verbose > 0:
            print(f"OpponentPoolCallback: Starting with {len(self.opponent_pool)} checkpoints in pool")

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Save final checkpoint
        self._save_checkpoint(is_final=True)

        if self.verbose > 0:
            print(f"OpponentPoolCallback: Training ended with {len(self.opponent_pool)} checkpoints")
            print(f"OpponentPoolCallback: Best Elo: {self.opponent_pool.best_elo():.1f}")

    def _save_checkpoint(self, is_initial: bool = False, is_final: bool = False) -> None:
        """Save a checkpoint to the opponent pool."""
        metadata = {
            "is_initial": is_initial,
            "is_final": is_final,
        }

        info = self.opponent_pool.save_checkpoint(
            model=self.model,
            step=self.num_timesteps,
            metadata=metadata,
        )

        if self.verbose > 0:
            prefix = "Initial" if is_initial else ("Final" if is_final else "Periodic")
            print(f"OpponentPoolCallback: {prefix} checkpoint saved: {info.checkpoint_id}")

        # Log to TensorBoard if available
        if self.logger is not None:
            self.logger.record("opponent_pool/size", len(self.opponent_pool))
            self.logger.record("opponent_pool/best_elo", self.opponent_pool.best_elo())
            self.logger.record("opponent_pool/elo_spread", self.opponent_pool.elo_spread())


class OpponentPoolEvalCallback(BaseCallback):
    """Callback for evaluating the current policy against the opponent pool.

    This callback periodically plays games against checkpoints from the pool
    and updates their win rates and Elo ratings. It provides actual evaluation
    data for PFSP matchmaking to work effectively.

    Example:
        >>> pool = OpponentPool(save_dir="checkpoints")
        >>> elo_tracker = EloTracker(save_path="elo_state.json")
        >>> callback = OpponentPoolEvalCallback(
        ...     opponent_pool=pool,
        ...     elo_tracker=elo_tracker,
        ...     env_factory=lambda: BusEnv(num_players=4),
        ...     eval_interval=100_000,
        ... )
        >>> model.learn(total_timesteps=1000000, callback=callback)
    """

    def __init__(
        self,
        opponent_pool: "OpponentPool",
        elo_tracker: Optional["EloTracker"] = None,
        env_factory: Optional[Callable[[], gym.Env]] = None,
        eval_interval: int = 100_000,
        n_eval_games: int = 5,
        max_opponents: int = 5,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: The opponent pool to evaluate against.
            elo_tracker: Optional Elo tracker for rating updates.
            env_factory: Factory function to create evaluation environments.
            eval_interval: Training steps between evaluations.
            n_eval_games: Number of games per opponent for evaluation.
            max_opponents: Maximum number of opponents to evaluate against.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.elo_tracker = elo_tracker
        self.env_factory = env_factory
        self.eval_interval = eval_interval
        self.n_eval_games = n_eval_games
        self.max_opponents = max_opponents
        self._last_eval_step = 0

        # Track evaluation results over time
        self._eval_history: list[dict] = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        current_step = self.num_timesteps

        if current_step - self._last_eval_step >= self.eval_interval:
            self._evaluate_against_pool()
            self._last_eval_step = current_step

        return True

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        # Initial evaluation if pool is not empty
        if len(self.opponent_pool) > 0:
            if self.verbose > 0:
                print("OpponentPoolEvalCallback: Running initial evaluation...")
            self._evaluate_against_pool()
        
        super()._on_training_start()

    def _evaluate_against_pool(self) -> None:
        """Evaluate current policy against pool checkpoints with actual games."""
        if len(self.opponent_pool) == 0:
            if self.verbose > 0:
                print("OpponentPoolEvalCallback: Pool is empty, skipping evaluation")
            return

        if self.verbose > 0:
            print(f"OpponentPoolEvalCallback: Evaluating against {min(self.max_opponents, len(self.opponent_pool))} opponents...")

        # If we have an env factory and elo tracker, run actual evaluations
        if self.env_factory is not None:
            results = self._run_evaluations()
            self._log_evaluation_results(results)
        else:
            # Fallback to basic logging
            self._log_basic_stats()

    def _run_evaluations(self) -> dict:
        """Run evaluation games against pool opponents."""
        from .multi_policy_env import MatchRunner

        runner = MatchRunner(
            env_factory=self.env_factory,
            elo_tracker=self.elo_tracker,
        )

        # Sample opponents to evaluate against
        opponents = self.opponent_pool.sample_opponents(
            n=min(self.max_opponents, len(self.opponent_pool)),
            method="uniform",
            allow_duplicates=False,
        )

        total_wins = 0
        total_games = 0
        results_by_opponent = {}

        for opponent_info in opponents:
            try:
                opponent_policy = self.opponent_pool.load_checkpoint(opponent_info)

                match_result = runner.run_match(
                    policy_a=self.model,
                    policy_b=opponent_policy,
                    checkpoint_id_a="__current__",
                    checkpoint_id_b=opponent_info.checkpoint_id,
                    n_games=self.n_eval_games,
                    randomize_seats=True,
                )

                total_wins += match_result["wins_a"]
                total_games += match_result["total_games"]

                # Store result
                results_by_opponent[opponent_info.checkpoint_id] = {
                    "win_rate": match_result["win_rate_a"],
                    "opponent_elo": opponent_info.elo,
                    "avg_score_diff": match_result["avg_score_a"] - match_result["avg_score_b"],
                }

                # Update opponent's win rate in pool (from opponent's perspective)
                self.opponent_pool.update_checkpoint_stats(
                    checkpoint_id=opponent_info.checkpoint_id,
                    win_rate=match_result["win_rate_b"],
                    games_played_delta=self.n_eval_games,
                )

                if self.verbose > 0:
                    print(f"  vs {opponent_info.checkpoint_id}: {match_result['win_rate_a']:.1%} win rate")

            except Exception as e:
                if self.verbose > 0:
                    print(f"  Failed to evaluate vs {opponent_info.checkpoint_id}: {e}")
                continue

        # Sync Elo from tracker to pool
        self.opponent_pool.sync_elo_from_tracker()

        overall_win_rate = total_wins / total_games if total_games > 0 else 0.5

        results = {
            "total_wins": total_wins,
            "total_games": total_games,
            "overall_win_rate": overall_win_rate,
            "opponents_evaluated": len(results_by_opponent),
            "results_by_opponent": results_by_opponent,
            "step": self.num_timesteps,
        }

        self._eval_history.append(results)

        return results

    def _log_evaluation_results(self, results: dict) -> None:
        """Log evaluation results to TensorBoard."""
        if self.logger is None:
            return

        self.logger.record("eval/overall_win_rate", results["overall_win_rate"])
        self.logger.record("eval/total_games", results["total_games"])
        self.logger.record("eval/opponents_evaluated", results["opponents_evaluated"])
        self.logger.record("opponent_pool/size", len(self.opponent_pool))
        self.logger.record("opponent_pool/best_elo", self.opponent_pool.best_elo())
        self.logger.record("opponent_pool/elo_spread", self.opponent_pool.elo_spread())

        if self.elo_tracker is not None:
            current_elo = self.elo_tracker.get_rating("__current__")
            self.logger.record("eval/current_elo", current_elo)
        
        # Log pool diversity
        self.logger.record("opponent_pool/elo_spread", self.opponent_pool.elo_spread())

    def _log_basic_stats(self) -> None:
        """Log basic pool statistics (fallback when no env factory)."""
        if self.verbose > 0:
            print(f"OpponentPoolEvalCallback: Pool has {len(self.opponent_pool)} checkpoints")
            if len(self.opponent_pool) > 0:
                best = self.opponent_pool.get_best_checkpoint()
                latest = self.opponent_pool.get_latest_checkpoint()
                print(f"  Best checkpoint: {best.checkpoint_id} (Elo: {best.elo:.1f})")
                print(f"  Latest checkpoint: {latest.checkpoint_id} (step: {latest.step})")

        if self.logger is not None:
            self.logger.record("opponent_pool/size", len(self.opponent_pool))
            self.logger.record("opponent_pool/best_elo", self.opponent_pool.best_elo())


class MultiPolicyTrainingCallback(BaseCallback):
    """Callback for managing multi-policy training with opponent sampling.

    This callback updates the MultiPolicyBusEnv wrapper with the current
    training policy and handles opponent assignment between episodes.
    """

    def __init__(
        self,
        opponent_pool: "OpponentPool",
        elo_tracker: Optional["EloTracker"] = None,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: Pool to sample opponents from.
            elo_tracker: Optional Elo tracker for rating management.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.elo_tracker = elo_tracker
        self._games_played = 0

    def _on_training_start(self) -> None:
        """Called at training start."""
        # Set up Elo tracker in pool if not already set
        if self.elo_tracker is not None and self.opponent_pool.elo_tracker is None:
            self.opponent_pool.elo_tracker = self.elo_tracker

        # Register current policy with Elo tracker
        if self.elo_tracker is not None:
            self.elo_tracker.register_checkpoint("__current__")

        if self.verbose > 0:
            print("MultiPolicyTrainingCallback: Training started")
            print(f"  Pool size: {len(self.opponent_pool)}")

    def _on_step(self) -> bool:
        """Called after each step."""
        # Check for completed episodes
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            if np.any(dones):
                self._games_played += np.sum(dones)

                # Sync Elo ratings periodically
                if self._games_played % 10 == 0:
                    self.opponent_pool.sync_elo_from_tracker()

        return True

    def _on_training_end(self) -> None:
        """Called at training end."""
        # Final sync
        self.opponent_pool.sync_elo_from_tracker()

        if self.verbose > 0:
            print(f"MultiPolicyTrainingCallback: Training ended")
            print(f"  Total games: {self._games_played}")
            print(f"  Pool size: {len(self.opponent_pool)}")
            print(f"  Best Elo: {self.opponent_pool.best_elo():.1f}")


class SyncPolicyCallback(BaseCallback):
    """Callback for synchronizing the live policy to disk for parallel workers.

    This bypasses the limitation of SubprocVecEnv not being able to share
    live model references. Workers will hot-reload this file.
    """

    def __init__(self, save_path: str, save_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_interval
        self._last_save_step = 0
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.save_interval:
            self.model.save(self.save_path)
            self._last_save_step = self.num_timesteps
        return True
    
    def _on_training_start(self) -> None:
        # Save immediately so workers finding it on first reset don't crash
        self.model.save(self.save_path)
