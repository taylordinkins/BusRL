"""Custom callbacks for RL training with opponent pool integration.

This module provides callbacks that integrate the opponent pool with
the training loop, enabling automatic checkpoint saving, pool management,
and evaluation against pool opponents with Elo rating updates.
"""

from __future__ import annotations

import os
import json
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .opponent_pool import OpponentPool
    from .elo_tracker import EloTracker

import multiprocessing

from rl.bus_env import BusEnv


class OpponentPoolCallback(BaseCallback):
    """Callback for managing the opponent pool during training.

    This callback:
    - Saves checkpoints to the opponent pool at regular intervals
    - Logs pool statistics to TensorBoard
    - Optionally maintains a "self-play checkpoint" for SubprocVecEnv workers

    Example:
        >>> pool = OpponentPool(save_dir="checkpoints")
        >>> callback = OpponentPoolCallback(pool, save_interval=50000)
        >>> model.learn(total_timesteps=1000000, callback=callback)
    """

    def __init__(
        self,
        opponent_pool: "OpponentPool",
        save_interval: int = 50_000,
        self_play_checkpoint_path: Optional[str] = None,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: The opponent pool to manage.
            save_interval: Training steps between checkpoint saves.
            self_play_checkpoint_path: If provided, also save a "current policy" checkpoint
                at this path for SubprocVecEnv workers to load when self_play_prob triggers.
                This enables SubprocVecEnv to use recent policy weights for self-play.
            verbose: Verbosity level (0 = no output, 1 = info).
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.save_interval = save_interval
        self.self_play_checkpoint_path = self_play_checkpoint_path
        self._last_save_step = 0

    @staticmethod
    def _is_main_process() -> bool:
        """Return True if this is the main process (not a SubprocVecEnv worker)."""
        return multiprocessing.current_process().name == "MainProcess"

    def _on_step(self) -> bool:
        """Called after each environment step."""
        current_step = self.num_timesteps

        if current_step - self._last_save_step >= self.save_interval:
            if self._is_main_process():
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

        # Save initial self-play checkpoint for SubprocVecEnv workers
        if self.self_play_checkpoint_path is not None:
            self._save_self_play_checkpoint()

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

        # Also update self-play checkpoint for SubprocVecEnv workers
        # Also update self-play checkpoint for SubprocVecEnv workers
        if self.self_play_checkpoint_path is not None:
            # Only save self-play checkpoint in main process
            if getattr(self.model, "env", None) is None or getattr(self.model.env, "num_envs", 1) == 1:
                self._save_self_play_checkpoint()


        # Log to TensorBoard if available
        if self.logger is not None:
            self.logger.record("opponent_pool/size", len(self.opponent_pool))
            self.logger.record("opponent_pool/best_elo", self.opponent_pool.best_elo())
            self.logger.record("opponent_pool/elo_spread", self.opponent_pool.elo_spread())

    def _save_self_play_checkpoint(self) -> None:
        """Save current policy to self-play checkpoint path.

        This allows SubprocVecEnv workers to load recent weights when
        self_play_prob triggers. Workers will pick up the new checkpoint
        at the start of their next episode.
        """
        if self.self_play_checkpoint_path is None:
            return

        # Save model (overwrites previous)
        self.model.save(self.self_play_checkpoint_path)

        if self.verbose > 0:
            print(f"OpponentPoolCallback: Self-play checkpoint updated at step {self.num_timesteps}")


class DiagnosticMaskingCallback(BaseCallback):
    """Lightweight diagnostics for action masks and masked distributions.

    Logs infrequent summaries to TensorBoard to catch numerical or masking issues
    without spamming training logs.
    """

    def __init__(
        self,
        log_interval: int = 100_000,
        max_samples: int = 256,
        prob_sum_tol: float = 5e-5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.max_samples = max_samples
        self.prob_sum_tol = prob_sum_tol
        self._last_log_step = 0

    def _sample_batch(self, obs, action_masks: np.ndarray):
        n = action_masks.shape[0]
        if n == 0:
            return None, None
        sample_n = min(self.max_samples, n)
        idx = np.random.choice(n, size=sample_n, replace=False)
        if isinstance(obs, dict):
            obs_sample = {k: v[idx] for k, v in obs.items()}
        else:
            obs_sample = obs[idx]
        mask_sample = action_masks[idx]
        return obs_sample, mask_sample

    def _to_tensor(self, obs_sample):
        if isinstance(obs_sample, dict):
            return {k: torch.as_tensor(v).to(self.model.device) for k, v in obs_sample.items()}
        return torch.as_tensor(obs_sample).to(self.model.device)

    def _on_rollout_end(self) -> None:
        if self.log_interval <= 0:
            return
        if self.num_timesteps - self._last_log_step < self.log_interval:
            return
        self._last_log_step = self.num_timesteps

        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        action_masks = getattr(rollout_buffer, "action_masks", None)
        if action_masks is None:
            return

        mask_np = np.asarray(action_masks)
        # Buffer layout is (buffer_size, n_envs, mask_dim); flatten to (N, mask_dim)
        if mask_np.ndim == 3:
            mask_np = mask_np.reshape(-1, mask_np.shape[-1])
        if mask_np.ndim != 2:
            return

        valid_counts = mask_np.sum(axis=1)
        mask_zero_count = int((valid_counts == 0).sum())

        if self.logger is not None:
            self.logger.record("debug/mask_valid_min", float(valid_counts.min()))
            self.logger.record("debug/mask_valid_max", float(valid_counts.max()))
            self.logger.record("debug/mask_valid_mean", float(valid_counts.mean()))
            self.logger.record("debug/mask_zero_count", mask_zero_count)

        # Flatten observations to match the mask layout (buffer_size, n_envs, ...) -> (N, ...)
        obs = rollout_buffer.observations
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            obs = obs.reshape(-1, *obs.shape[2:])
        elif isinstance(obs, dict):
            obs = {k: (v.reshape(-1, *v.shape[2:]) if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in obs.items()}

        obs_sample, mask_sample = self._sample_batch(obs, mask_np)
        if obs_sample is None:
            return

        try:
            with torch.no_grad():
                obs_t = self._to_tensor(obs_sample)
                mask_t = torch.as_tensor(mask_sample, dtype=torch.bool).to(self.model.device)
                dist = self.model.policy.get_distribution(obs_t, action_masks=mask_t)
                probs = dist.distribution.probs
                probs_sum = probs.sum(dim=-1)

                probs_sum_min = float(probs_sum.min().item())
                probs_sum_max = float(probs_sum.max().item())
                probs_sum_dev = float((probs_sum - 1.0).abs().max().item())
                probs_sum_bad = int(((probs_sum - 1.0).abs() > self.prob_sum_tol).sum().item())

                nonfinite_logits = 0
                logits = getattr(dist.distribution, "logits", None)
                if logits is not None:
                    nonfinite_logits = int((~torch.isfinite(logits)).sum().item())

            if self.logger is not None:
                self.logger.record("debug/probs_sum_min", probs_sum_min)
                self.logger.record("debug/probs_sum_max", probs_sum_max)
                self.logger.record("debug/probs_sum_dev_max", probs_sum_dev)
                self.logger.record("debug/probs_sum_bad", probs_sum_bad)
                self.logger.record("debug/nonfinite_logits", nonfinite_logits)
        except Exception as exc:
            if self.verbose > 0:
                print(f"DiagnosticMaskingCallback warning: {exc}")

    def _on_step(self) -> bool:
        # Required by BaseCallback; no per-step work needed.
        return True


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
        best_model_save_path: Optional[str] = None,
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
            best_model_save_path: If provided, save the current model here whenever
                overall pool win rate improves. A best_model_info.json is written
                alongside with win_rate, elo, and step for tracking across generations.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.elo_tracker = elo_tracker
        self.env_factory = env_factory
        self.eval_interval = eval_interval
        self.n_eval_games = n_eval_games
        self.max_opponents = max_opponents
        self.best_model_save_path = best_model_save_path
        self._last_eval_step = 0
        self._best_win_rate = -1.0

        # Track evaluation results over time
        self._eval_history: list[dict] = []

        print("Will play {} opponent games for eval".format(self.n_eval_games))

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
            self._maybe_save_best_model(results)
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

    def _maybe_save_best_model(self, results: dict) -> None:
        """Save current model and metadata if pool win rate improved."""
        if self.best_model_save_path is None:
            return

        win_rate = results["overall_win_rate"]
        if win_rate <= self._best_win_rate:
            return

        self._best_win_rate = win_rate

        os.makedirs(self.best_model_save_path, exist_ok=True)
        self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        current_elo = self.elo_tracker.get_rating("__current__") if self.elo_tracker else 1500.0
        info = {
            "win_rate": win_rate,
            "elo": current_elo,
            "step": self.num_timesteps,
            "opponents_evaluated": results["opponents_evaluated"],
            "total_games": results["total_games"],
        }
        with open(os.path.join(self.best_model_save_path, "best_model_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        if self.logger is not None:
            self.logger.record("eval/best_pool_win_rate", win_rate)
            self.logger.record("eval/best_pool_elo", current_elo)

        if self.verbose > 0:
            print(f"OpponentPoolEvalCallback: New best model saved "
                  f"(win_rate={win_rate:.3f}, elo={current_elo:.1f}, step={self.num_timesteps})")

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
        dones = self.locals.get("dones")
        if dones is not None:
            dones = np.array(dones, dtype=bool)
            if dones.any():
                self._games_played += dones.sum()

                # Sync Elo ratings periodically
                if self._games_played % 10 == 0:
                    # Only sync from main process
                    if getattr(self.model, "env", None) is None or getattr(self.model.env, "num_envs", 1) == 1:
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

class TrueEpisodeLengthCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Get _episode_lengths from all envs
        env_episode_lengths_list = self.training_env.get_attr("_episode_lengths")  # list of lists

        for env_episode_lengths in env_episode_lengths_list:
            while env_episode_lengths:
                ep_len = env_episode_lengths.pop(0)
                if self.logger:
                    self.logger.record("rollout/ep_len_true", ep_len)

        return True
