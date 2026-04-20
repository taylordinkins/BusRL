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
from sb3_contrib.common.maskable.utils import get_action_masks
from collections import defaultdict

if TYPE_CHECKING:
    from .opponent_pool import OpponentPool
    from .elo_tracker import EloTracker

import multiprocessing

from rl.bus_env import BusEnv
from rl.hierarchical_action_space import HeadId


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
        min_games_for_live_elo: int = 0,
        recenter_interval_steps: int = 1_000_000,
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
        self.min_games_for_live_elo = min_games_for_live_elo
        self.recenter_interval_steps = max(0, int(recenter_interval_steps))
        self._last_save_step = 0
        self._last_recenter_step = 0

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

        if (
            self.recenter_interval_steps > 0
            and current_step - self._last_recenter_step >= self.recenter_interval_steps
        ):
            tracker = self.opponent_pool.elo_tracker
            if tracker is not None and hasattr(tracker, "recenter"):
                if self._is_main_process():
                    tracker.recenter()
                    if self.verbose > 0:
                        print(
                            f"OpponentPoolCallback: Recentered OpenSkill mus at step {current_step}"
                        )
                self._last_recenter_step = current_step

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

        elo_value = None
        sigma_value = None
        tracker = self.opponent_pool.elo_tracker
        if tracker is not None:
            # Only use live rating after enough games to be meaningful
            games_played = 0
            if hasattr(tracker, "get_games_played"):
                games_played = tracker.get_games_played("__current__")
            if games_played >= self.min_games_for_live_elo:
                elo_value = tracker.get_rating("__current__")
                metadata["games_played"] = games_played
                if hasattr(tracker, "get_sigma"):
                    sigma_value = tracker.get_sigma("__current__")
                    metadata["sigma"] = sigma_value

        info = self.opponent_pool.save_checkpoint(
            model=self.model,
            step=self.num_timesteps,
            elo=elo_value,
            sigma=sigma_value,
            metadata=metadata,
        )

        if self.verbose > 0:
            prefix = "Initial" if is_initial else ("Final" if is_final else "Periodic")
            print(f"OpponentPoolCallback: {prefix} checkpoint saved: {info.checkpoint_id}")

        # Also update self-play checkpoint for SubprocVecEnv workers
        if self.self_play_checkpoint_path is not None:
            if self._is_main_process():
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

        # Save model atomically to avoid workers reading partial files
        self._save_model_atomic(self.model, self.self_play_checkpoint_path)

        if self.verbose > 0:
            print(f"OpponentPoolCallback: Self-play checkpoint updated at step {self.num_timesteps}")

    @staticmethod
    def _save_model_atomic(model, target_path: str) -> None:
        """Atomically save an SB3 model by writing a temp file then renaming."""
        import tempfile

        target_path = str(target_path)
        if target_path.endswith(".zip"):
            final_path = target_path
        else:
            final_path = target_path + ".zip"

        directory = os.path.dirname(final_path) or "."
        os.makedirs(directory, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix=".self_play_", suffix=".zip", dir=directory)
        os.close(fd)
        saved_path = tmp_path
        try:
            model.save(tmp_path)
            # Some SB3 versions append .zip if missing; be defensive.
            if not os.path.exists(saved_path) and os.path.exists(tmp_path + ".zip"):
                saved_path = tmp_path + ".zip"
            os.replace(saved_path, final_path)
        finally:
            for path in (tmp_path, tmp_path + ".zip"):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass


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
        skill_tracking: str = "elo",
        skill_temperature: float = 3.0,
        verbose: int = 0,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: The opponent pool to evaluate against.
            elo_tracker: Optional EloTracker or OpenSkillTracker for rating updates.
            env_factory: Factory function to create evaluation environments.
            eval_interval: Training steps between evaluations.
            n_eval_games: Number of games per opponent for evaluation.
            max_opponents: Maximum number of opponents to evaluate against.
            best_model_save_path: If provided, save the current model here whenever
                overall pool win rate improves. A best_model_info.json is written
                alongside with win_rate, elo/mu, and step for tracking across generations.
            skill_tracking: "elo" or "openskill".  Controls eval loop structure
                and TensorBoard label conventions.
            skill_temperature: Softmax temperature for opponent sampling in
                openskill eval mode.  Only used when skill_tracking=="openskill".
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
        self.skill_tracking = skill_tracking
        self.skill_temperature = skill_temperature
        self._last_eval_step = 0
        self._best_win_rate = -1.0
        self._best_skill = float("-inf")

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
        """Dispatch to the mode-appropriate evaluation loop."""
        if self.skill_tracking == "openskill":
            return self._run_evaluations_openskill()
        return self._run_evaluations_elo()

    def _run_evaluations_elo(self) -> dict:
        """Run evaluation games against pool opponents (Elo / 2-policy mode)."""
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

        total_points = 0.0
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

                total_points += match_result.get(
                    "points_a",
                    match_result["wins_a"] + 0.5 * match_result.get("draws", 0),
                )
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

        overall_win_rate = total_points / total_games if total_games > 0 else 0.5

        results = {
            "total_wins": total_points,
            "total_games": total_games,
            "overall_win_rate": overall_win_rate,
            "opponents_evaluated": len(results_by_opponent),
            "results_by_opponent": results_by_opponent,
            "step": self.num_timesteps,
        }

        self._eval_history.append(results)

        return results

    def _run_evaluations_openskill(self) -> dict:
        """Run true N-player evaluation games (OpenSkill mode).

        Each game is current + 3 opponents sampled via temperature-weighted
        softmax.  Total games = max_opponents * n_eval_games (same budget as
        Elo mode).  Ratings are updated via the native Plackett-Luce model
        after every game.
        """
        from .multi_policy_env import MatchRunner

        runner = MatchRunner(
            env_factory=self.env_factory,
            elo_tracker=self.elo_tracker,
        )

        total_games_budget = self.max_opponents * self.n_eval_games
        total_points = 0.0
        total_games = 0
        results_by_opponent: dict[str, dict] = {}

        # Cache loaded opponent policies to avoid repeated disk reads
        policy_cache: dict = {}

        for _ in range(total_games_budget):
            pool_size = len(self.opponent_pool)
            if pool_size == 0:
                break

            # Sample 3 opponents
            if pool_size >= 3:
                opponents = self.opponent_pool.sample_opponents_temperature_weighted(
                    n=3, temperature=self.skill_temperature, tracker=self.elo_tracker
                )
            else:
                # Pool < 3: fill with replacement from what's available
                opponents = []
                for _ in range(3):
                    opp = self.opponent_pool.sample_opponent(method="uniform")
                    if opp:
                        opponents.append(opp)

            if not opponents:
                break

            # Load opponent policies (cached)
            opp_policies = []
            opp_ids = []
            for opp_info in opponents:
                if opp_info.checkpoint_id not in policy_cache:
                    try:
                        policy_cache[opp_info.checkpoint_id] = self.opponent_pool.load_checkpoint(opp_info)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"  Failed to load opponent {opp_info.checkpoint_id}: {e}")
                        continue
                if opp_info.checkpoint_id in policy_cache:
                    opp_policies.append(policy_cache[opp_info.checkpoint_id])
                    opp_ids.append(opp_info.checkpoint_id)

            if not opp_policies:
                continue

            # Pad to 3 if fewer loaded (reuse last available)
            while len(opp_policies) < 3:
                opp_policies.append(opp_policies[-1])
                opp_ids.append(opp_ids[-1])

            # 4-player game: current + 3 opponents
            policies = [self.model] + opp_policies
            checkpoint_ids = ["__current__"] + opp_ids

            try:
                game_results = runner.run_multiplayer_match(policies, checkpoint_ids, n_games=1)
                result = game_results[0]
            except Exception as e:
                if self.verbose > 0:
                    print(f"  Multiplayer match failed: {e}")
                continue

            avg_scores = result["avg_scores"]

            # Update OpenSkill ratings with full 4-player result
            game_ids = list(avg_scores.keys())
            game_scores = [avg_scores[pid] for pid in game_ids]
            self.elo_tracker.update_ratings_multiplayer(game_ids, game_scores)

            # Draw-aware points: win=1.0, draw for top=0.5, loss=0.0
            current_score = avg_scores.get("__current__", 0)
            max_score = max(avg_scores.values())
            top_count = sum(1 for s in avg_scores.values() if s == max_score)
            if current_score == max_score and top_count == 1:
                total_points += 1.0
            elif current_score == max_score and top_count > 1:
                total_points += 0.5
            total_games += 1

            # Track per-opponent stats
            for opp_id in opp_ids:
                if opp_id not in results_by_opponent:
                    results_by_opponent[opp_id] = {
                        "games": 0,
                        "opponent_elo": self.elo_tracker.get_rating(opp_id),
                    }
                results_by_opponent[opp_id]["games"] += 1

        # Sync mu values from tracker to pool checkpoints
        self.opponent_pool.sync_elo_from_tracker()

        overall_win_rate = total_points / total_games if total_games > 0 else 0.5

        if self.verbose > 0:
            for opp_id, info in sorted(
                results_by_opponent.items(),
                key=lambda x: -x[1].get("opponent_elo", 0),
            ):
                print(f"  vs {opp_id}: {info['games']} games, mu={self.elo_tracker.get_rating(opp_id):.1f}")

        return {
            "total_wins": total_points,
            "total_games": total_games,
            "overall_win_rate": overall_win_rate,
            "opponents_evaluated": len(results_by_opponent),
            "results_by_opponent": results_by_opponent,
            "step": self.num_timesteps,
        }

    def _log_evaluation_results(self, results: dict) -> None:
        """Log evaluation results to TensorBoard (mode-aware labels)."""
        if self.logger is None:
            return

        # Labels common to both modes
        self.logger.record("eval/overall_win_rate", results["overall_win_rate"])
        self.logger.record("eval/total_games", results["total_games"])
        self.logger.record("eval/opponents_evaluated", results["opponents_evaluated"])
        self.logger.record("opponent_pool/size", len(self.opponent_pool))

        if self.skill_tracking == "openskill":
            if self.elo_tracker is not None:
                self.logger.record("eval/current_mu", self.elo_tracker.get_rating("__current__"))
            self.logger.record("opponent_pool/best_mu", self.opponent_pool.best_elo())
            self.logger.record("opponent_pool/mu_spread", self.opponent_pool.elo_spread())
        else:
            self.logger.record("opponent_pool/best_elo", self.opponent_pool.best_elo())
            self.logger.record("opponent_pool/elo_spread", self.opponent_pool.elo_spread())
            if self.elo_tracker is not None:
                self.logger.record("eval/current_elo", self.elo_tracker.get_rating("__current__"))

    def _maybe_save_best_model(self, results: dict) -> None:
        """Save current model and metadata if skill rating (mu/Elo) improved.

        Falls back to win-rate gating if no tracker is available.
        """
        if self.best_model_save_path is None:
            return

        win_rate = results["overall_win_rate"]

        # Prefer skill rating (mu/Elo) as the primary selection criterion
        current_skill = None
        if self.elo_tracker is not None:
            current_skill = self.elo_tracker.get_rating("__current__")

        if current_skill is not None:
            if current_skill <= self._best_skill:
                return
            self._best_skill = current_skill
        else:
            # Fallback to win-rate gating when tracker is missing
            if win_rate <= self._best_win_rate:
                return
            self._best_win_rate = win_rate

        self._best_win_rate = max(self._best_win_rate, win_rate)

        os.makedirs(self.best_model_save_path, exist_ok=True)
        self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        info: dict = {
            "win_rate": win_rate,
            "step": self.num_timesteps,
            "opponents_evaluated": results["opponents_evaluated"],
            "total_games": results["total_games"],
        }

        if self.skill_tracking == "openskill":
            current_mu = current_skill if current_skill is not None else 1500.0
            info["mu"] = current_mu
            info["skill_tracking"] = "openskill"
            if self.elo_tracker and hasattr(self.elo_tracker, "get_sigma"):
                info["sigma"] = self.elo_tracker.get_sigma("__current__")
        else:
            current_elo = current_skill if current_skill is not None else 1500.0
            info["elo"] = current_elo

        with open(os.path.join(self.best_model_save_path, "best_model_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        if self.logger is not None:
            self.logger.record("eval/best_pool_win_rate", win_rate)
            if self.skill_tracking == "openskill":
                self.logger.record("eval/best_pool_mu", info["mu"])
            else:
                self.logger.record("eval/best_pool_elo", info["elo"])

        if self.verbose > 0:
            if self.skill_tracking == "openskill":
                print(f"OpponentPoolEvalCallback: New best model saved "
                      f"(win_rate={win_rate:.3f}, mu={info['mu']:.1f}, step={self.num_timesteps})")
            else:
                print(f"OpponentPoolEvalCallback: New best model saved "
                      f"(win_rate={win_rate:.3f}, elo={info['elo']:.1f}, step={self.num_timesteps})")

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

                # Sync Elo ratings periodically (main process only)
                if self._games_played % 10 == 0:
                    if multiprocessing.current_process().name == "MainProcess":
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


class HeadUsageCallback(BaseCallback):
    """Log per-head usage counts to TensorBoard."""

    def __init__(self, log_interval_steps: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval_steps = max(0, int(log_interval_steps))
        self._counts = {head: 0 for head in HeadId}
        self._last_log_step = 0
        self._label_map = {
            HeadId.SETUP_BUILDINGS: "setup_bld",
            HeadId.SETUP_RAILS_FORWARD: "setup_rf",
            HeadId.SETUP_RAILS_REVERSE: "setup_rr",
            HeadId.CHOOSING_ACTIONS: "choose",
            HeadId.RESOLVE_LINE_EXPANSION: "lineexp",
            HeadId.RESOLVE_PASSENGERS: "passeng",
            HeadId.RESOLVE_BUILDINGS: "build",
            HeadId.RESOLVE_TIME_CLOCK: "clock",
            HeadId.RESOLVE_VRROOMM_PASSENGER: "vr_pax",
            HeadId.RESOLVE_VRROOMM_DEST: "vr_dst",
        }
        self._marker_area_counts: dict[str, int] = {}
        self._opportunity_area_counts: dict[str, int] = {}
        self._reward_by_area: dict[str, float] = {}
        self._waste_area_counts: dict[str, int] = {}
        self._resolution_total_counts: dict[str, int] = {}
        self._waste_total: int = 0
        self._seen_waste_rounds: set[tuple[int, int, int, str]] = set()
        self._seen_opportunity_slots: set[tuple[int, int, int, str, str, int]] = set()
        self._episode_counters = defaultdict(int)
        self._internal_steps_total: int = 0

    def _on_training_start(self) -> None:
        labels = ", ".join(
            f"h{head.value}={self._label_map.get(head, head.name)}"
            for head in HeadId
        )
        print(f"Head usage labels: {labels}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos:
            for env_idx, info in enumerate(infos):
                episode_id = int(self._episode_counters[env_idx])
                self._internal_steps_total += int(info.get("telemetry_internal_steps", 1))

                telemetry_head_counts = info.get("telemetry_head_counts")
                if telemetry_head_counts:
                    for head_key, count in telemetry_head_counts.items():
                        try:
                            head_id = HeadId(int(head_key))
                        except (TypeError, ValueError):
                            continue
                        self._counts[head_id] += int(count)
                else:
                    head_id = info.get("action_head_id", info.get("head_id"))
                    if head_id is not None:
                        if isinstance(head_id, str):
                            try:
                                head_id = HeadId[head_id]
                            except KeyError:
                                head_id = None
                        elif isinstance(head_id, int):
                            try:
                                head_id = HeadId(head_id)
                            except ValueError:
                                head_id = None
                    if isinstance(head_id, HeadId):
                        self._counts[head_id] += 1

                telemetry_marker_counts = info.get("telemetry_marker_counts")
                if telemetry_marker_counts:
                    for area_key, count in telemetry_marker_counts.items():
                        self._marker_area_counts[area_key] = (
                            self._marker_area_counts.get(area_key, 0) + int(count)
                        )
                else:
                    placed_area = info.get("placed_marker_area")
                    if placed_area:
                        self._marker_area_counts[placed_area] = (
                            self._marker_area_counts.get(placed_area, 0) + 1
                        )

                telemetry_opportunity_counts = info.get("telemetry_opportunity_counts")
                if telemetry_opportunity_counts:
                    for area_key, count in telemetry_opportunity_counts.items():
                        self._opportunity_area_counts[area_key] = (
                            self._opportunity_area_counts.get(area_key, 0) + int(count)
                        )
                else:
                    resolution_area = info.get("resolution_area")
                    resolution_slot_label = info.get("resolution_slot_label")
                    resolution_slot_player = info.get("resolution_slot_player")
                    round_num = info.get("round")
                    valid_action_count = int(info.get("valid_action_count", 0) or 0)
                    if (
                        resolution_area
                        and resolution_slot_label is not None
                        and resolution_slot_player is not None
                        and round_num is not None
                        and valid_action_count > 0
                    ):
                        token = (
                            env_idx,
                            episode_id,
                            int(round_num),
                            str(resolution_area),
                            str(resolution_slot_label),
                            int(resolution_slot_player),
                        )
                        if token not in self._seen_opportunity_slots:
                            self._seen_opportunity_slots.add(token)
                            self._opportunity_area_counts[str(resolution_area)] = (
                                self._opportunity_area_counts.get(str(resolution_area), 0)
                                + 1
                            )

                telemetry_reward_by_area = info.get("telemetry_reward_by_area")
                if telemetry_reward_by_area:
                    for area_key, reward_sum in telemetry_reward_by_area.items():
                        self._reward_by_area[area_key] = float(
                            self._reward_by_area.get(area_key, 0.0)
                        ) + float(reward_sum)
                else:
                    head_id = info.get("action_head_id", info.get("head_id"))
                    reward_area = None
                    try:
                        head_int = int(head_id) if head_id is not None else None
                    except (TypeError, ValueError):
                        head_int = None
                    if head_int == 4:
                        reward_area = "line_expansion"
                    elif head_int == 5:
                        reward_area = "passengers"
                    elif head_int == 6:
                        reward_area = "buildings"
                    elif head_int == 7:
                        reward_area = "time_clock"
                    elif head_int in (8, 9):
                        reward_area = "vrroomm"
                    rewards = self.locals.get("rewards")
                    if reward_area is not None and rewards is not None and env_idx < len(np.atleast_1d(rewards)):
                        reward_value = float(np.atleast_1d(rewards)[env_idx])
                        self._reward_by_area[reward_area] = float(
                            self._reward_by_area.get(reward_area, 0.0)
                        ) + reward_value

                telemetry_wasted_counts = info.get("telemetry_wasted_counts")
                telemetry_resolution_totals = info.get("telemetry_resolution_totals")
                if telemetry_wasted_counts:
                    for area_key, count in telemetry_wasted_counts.items():
                        c = int(count)
                        self._waste_area_counts[area_key] = (
                            self._waste_area_counts.get(area_key, 0) + c
                        )
                        self._waste_total += c
                    if telemetry_resolution_totals:
                        for area_key, total in telemetry_resolution_totals.items():
                            self._resolution_total_counts[area_key] = (
                                self._resolution_total_counts.get(area_key, 0) + int(total)
                            )
                else:
                    waste_by_area = info.get("resolution_waste_by_area")
                    round_num = info.get("round")
                    if waste_by_area and round_num is not None:
                        for area_key, stats in waste_by_area.items():
                            token = (env_idx, episode_id, int(round_num), area_key)
                            if token in self._seen_waste_rounds:
                                continue
                            self._seen_waste_rounds.add(token)
                            wasted = int(stats.get("wasted", 0))
                            self._waste_area_counts[area_key] = (
                                self._waste_area_counts.get(area_key, 0) + wasted
                            )
                            self._resolution_total_counts[area_key] = (
                                self._resolution_total_counts.get(area_key, 0)
                                + int(stats.get("total", 0))
                            )
                            self._waste_total += wasted
        if dones is not None:
            done_arr = np.atleast_1d(dones)
            for env_idx, done in enumerate(done_arr):
                if done:
                    self._episode_counters[env_idx] += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is None:
            return
        if self.log_interval_steps and (self.num_timesteps - self._last_log_step < self.log_interval_steps):
            return
        self._last_log_step = self.num_timesteps

        total = sum(self._counts.values())
        for head, count in self._counts.items():
            short = f"h{head.value}"
            label = self._label_map.get(head, head.name)
            key = f"{short}_{label}"
            self.logger.record(f"rollout/head_usage/{key}", count)
            if total > 0:
                self.logger.record(f"rollout/head_usage_pct/{key}", count / total)

        for area_key, count in self._marker_area_counts.items():
            self.logger.record(f"rollout/marker_placed/{area_key}", count)

        tracked_areas = set(self._marker_area_counts.keys()) | set(self._opportunity_area_counts.keys())
        for area_key in sorted(tracked_areas):
            markers = int(self._marker_area_counts.get(area_key, 0))
            opportunities = int(self._opportunity_area_counts.get(area_key, 0))
            self.logger.record(f"rollout/marker_opportunities/{area_key}", opportunities)
            if markers > 0:
                self.logger.record(
                    f"rollout/marker_to_opportunity_ratio/{area_key}",
                    opportunities / markers,
                )

        reward_areas = set(self._reward_by_area.keys()) | set(self._marker_area_counts.keys())
        for area_key in sorted(reward_areas):
            markers = int(self._marker_area_counts.get(area_key, 0))
            reward_sum = float(self._reward_by_area.get(area_key, 0.0))
            self.logger.record(f"rollout/reward_by_marker_area/{area_key}", reward_sum)
            if markers > 0:
                self.logger.record(
                    f"rollout/reward_yield_per_marker/{area_key}",
                    reward_sum / markers,
                )

        for area_key, count in self._waste_area_counts.items():
            self.logger.record(f"rollout/wasted_markers/{area_key}", count)
            total_markers = self._resolution_total_counts.get(area_key, 0)
            self.logger.record(f"rollout/resolution_markers_total/{area_key}", total_markers)
            if total_markers > 0:
                self.logger.record(
                    f"rollout/wasted_marker_rate/{area_key}",
                    count / total_markers,
                )
            actionable = max(0, int(total_markers) - int(count))
            opportunities = int(self._opportunity_area_counts.get(area_key, 0))
            avoidable_est = max(0, actionable - opportunities)
            self.logger.record(
                f"rollout/waste_unavoidable/{area_key}",
                int(count),
            )
            self.logger.record(
                f"rollout/waste_actionable_marker_pool/{area_key}",
                actionable,
            )
            self.logger.record(
                f"rollout/waste_avoidable_est/{area_key}",
                avoidable_est,
            )
            if actionable > 0:
                self.logger.record(
                    f"rollout/waste_avoidable_rate_est/{area_key}",
                    avoidable_est / actionable,
                )
        self.logger.record("rollout/wasted_markers_total", self._waste_total)
        self.logger.record("rollout/internal_steps_total", self._internal_steps_total)

        self._counts = {head: 0 for head in HeadId}
        self._marker_area_counts = {}
        self._opportunity_area_counts = {}
        self._reward_by_area = {}
        self._waste_area_counts = {}
        self._resolution_total_counts = {}
        self._waste_total = 0
        self._seen_waste_rounds = set()
        self._seen_opportunity_slots = set()
        self._internal_steps_total = 0


class EvalStatsCallback(BaseCallback):
    """Run periodic eval episodes and log score diff + head usage."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        debug_waste_log_path: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = bool(deterministic)
        # When provided, write per-slot resolution detail for every eval episode
        # to this path (JSON lines, one object per resolution event).
        self.debug_waste_log_path = debug_waste_log_path
        self._label_map = {
            HeadId.SETUP_BUILDINGS: "setup_bld",
            HeadId.SETUP_RAILS_FORWARD: "setup_rf",
            HeadId.SETUP_RAILS_REVERSE: "setup_rr",
            HeadId.CHOOSING_ACTIONS: "choose",
            HeadId.RESOLVE_LINE_EXPANSION: "lineexp",
            HeadId.RESOLVE_PASSENGERS: "passeng",
            HeadId.RESOLVE_BUILDINGS: "build",
            HeadId.RESOLVE_TIME_CLOCK: "clock",
            HeadId.RESOLVE_VRROOMM_PASSENGER: "vr_pax",
            HeadId.RESOLVE_VRROOMM_DEST: "vr_dst",
        }

    def _run_eval(self) -> None:
        score_diffs: list[float] = []
        head_counts = {head: 0 for head in HeadId}
        total_steps = 0
        marker_area_counts: dict[str, int] = {}
        opportunity_area_counts: dict[str, int] = {}
        reward_by_area: dict[str, float] = {}
        waste_area_counts: dict[str, int] = {}
        resolution_total_counts: dict[str, int] = {}
        waste_total = 0
        seen_waste_rounds: set[tuple[int, str]] = set()
        seen_opportunity_slots: set[tuple[int, int, str, str, int]] = set()
        _debug_records: list[dict] = []
        _slot_label_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
        _moob_areas = {"line_expansion", "passengers", "buildings"}

        for episode_idx in range(self.n_eval_episodes):
            reset_out = self.eval_env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, _ = reset_out
            else:
                obs = reset_out
            terminated = False
            truncated = False

            while not (terminated or truncated):
                masks = get_action_masks(self.eval_env)
                action, _ = self.model.predict(
                    obs, action_masks=masks, deterministic=self.deterministic
                )
                step_out = self.eval_env.step(action)
                if len(step_out) == 4:
                    obs, rewards, dones, infos = step_out
                    truncs = [False for _ in np.atleast_1d(dones)]
                else:
                    obs, rewards, dones, truncs, infos = step_out
                terminated = bool(np.atleast_1d(dones)[0])
                truncated = bool(np.atleast_1d(truncs)[0])

                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                head_id = info.get("action_head_id", info.get("head_id"))
                head_int = None
                try:
                    if head_id is not None:
                        head_int = int(head_id)
                except (TypeError, ValueError):
                    head_int = None
                if head_id is not None:
                    if isinstance(head_id, str):
                        try:
                            head_id = HeadId[head_id]
                        except KeyError:
                            head_id = None
                    elif isinstance(head_id, int):
                        try:
                            head_id = HeadId(head_id)
                        except ValueError:
                            head_id = None
                if isinstance(head_id, HeadId):
                    head_counts[head_id] += 1
                placed_area = info.get("placed_marker_area")
                if placed_area:
                    marker_area_counts[placed_area] = (
                        marker_area_counts.get(placed_area, 0) + 1
                    )

                resolution_area = info.get("resolution_area")
                resolution_slot_label = info.get("resolution_slot_label")
                resolution_slot_player = info.get("resolution_slot_player")
                round_num = info.get("round")
                valid_action_count = int(info.get("valid_action_count", 0) or 0)
                if (
                    resolution_area
                    and resolution_slot_label is not None
                    and resolution_slot_player is not None
                    and round_num is not None
                    and valid_action_count > 0
                ):
                    token = (
                        int(episode_idx),
                        int(round_num),
                        str(resolution_area),
                        str(resolution_slot_label),
                        int(resolution_slot_player),
                    )
                    if token not in seen_opportunity_slots:
                        seen_opportunity_slots.add(token)
                        opportunity_area_counts[str(resolution_area)] = (
                            opportunity_area_counts.get(str(resolution_area), 0) + 1
                        )

                reward_area = None
                if head_int == 4:
                    reward_area = "line_expansion"
                elif head_int == 5:
                    reward_area = "passengers"
                elif head_int == 6:
                    reward_area = "buildings"
                elif head_int == 7:
                    reward_area = "time_clock"
                elif head_int in (8, 9):
                    reward_area = "vrroomm"
                if reward_area is not None:
                    reward_value = float(np.atleast_1d(rewards)[0])
                    reward_by_area[reward_area] = float(
                        reward_by_area.get(reward_area, 0.0)
                    ) + reward_value

                waste_by_area = info.get("resolution_waste_by_area")
                if waste_by_area and round_num is not None:
                    for area_key, stats in waste_by_area.items():
                        token = (int(episode_idx), int(round_num), area_key)
                        if token in seen_waste_rounds:
                            continue
                        seen_waste_rounds.add(token)
                        wasted = int(stats.get("wasted", 0))
                        waste_area_counts[area_key] = (
                            waste_area_counts.get(area_key, 0) + wasted
                        )
                        resolution_total_counts[area_key] = (
                            resolution_total_counts.get(area_key, 0)
                            + int(stats.get("total", 0))
                        )
                        waste_total += wasted

                # Debug: per-slot resolution detail for waste metric verification
                if (
                    self.debug_waste_log_path
                    and resolution_area in _moob_areas
                    and resolution_slot_label is not None
                    and resolution_slot_player is not None
                    and round_num is not None
                ):
                    slot_idx = _slot_label_to_index.get(str(resolution_slot_label).upper(), -1)
                    area_waste_stats = (waste_by_area or {}).get(str(resolution_area), {})
                    max_buses = area_waste_stats.get("max_buses", -1)
                    is_type1_wasted = (
                        slot_idx >= 0 and max_buses >= 0 and (max_buses - slot_idx) <= 0
                    )
                    _debug_records.append({
                        "step": self.num_timesteps,
                        "episode": episode_idx,
                        "round": int(round_num),
                        "area": str(resolution_area),
                        "slot": str(resolution_slot_label),
                        "slot_index": slot_idx,
                        "player": resolution_slot_player,
                        "max_buses": max_buses,
                        "valid_action_count": valid_action_count,
                        "is_type1_wasted": is_type1_wasted,
                        "is_type2_candidate": (
                            not is_type1_wasted and valid_action_count == 0
                        ),
                    })

                total_steps += 1

            # End of episode: compute score diff using score - time_stones
            final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
            scores = final_info.get("scores", {})
            time_stones = final_info.get("time_stones", {})
            if scores:
                final_scores = [
                    scores[p_id] - time_stones.get(p_id, 0) for p_id in scores.keys()
                ]
                final_scores.sort(reverse=True)
                if len(final_scores) >= 2:
                    score_diffs.append(final_scores[0] - final_scores[1])
                else:
                    score_diffs.append(0.0)

        # Write per-slot resolution debug records if requested
        if self.debug_waste_log_path and _debug_records:
            os.makedirs(os.path.dirname(self.debug_waste_log_path) or ".", exist_ok=True)
            with open(self.debug_waste_log_path, "a") as _f:
                for _rec in _debug_records:
                    _f.write(json.dumps(_rec) + "\n")

        if self.logger is None:
            return

        if score_diffs:
            self.logger.record("eval/score_diff_top2", float(np.mean(score_diffs)))

        for head, count in head_counts.items():
            short = f"h{head.value}"
            label = self._label_map.get(head, head.name)
            key = f"{short}_{label}"
            self.logger.record(f"eval/head_usage/{key}", count)
            if total_steps > 0:
                self.logger.record(f"eval/head_usage_pct/{key}", count / total_steps)

        for area_key, count in marker_area_counts.items():
            self.logger.record(f"eval/marker_placed/{area_key}", count)

        tracked_areas = set(marker_area_counts.keys()) | set(opportunity_area_counts.keys())
        for area_key in sorted(tracked_areas):
            markers = int(marker_area_counts.get(area_key, 0))
            opportunities = int(opportunity_area_counts.get(area_key, 0))
            self.logger.record(f"eval/marker_opportunities/{area_key}", opportunities)
            if markers > 0:
                self.logger.record(
                    f"eval/marker_to_opportunity_ratio/{area_key}",
                    opportunities / markers,
                )

        reward_areas = set(reward_by_area.keys()) | set(marker_area_counts.keys())
        for area_key in sorted(reward_areas):
            markers = int(marker_area_counts.get(area_key, 0))
            reward_sum = float(reward_by_area.get(area_key, 0.0))
            self.logger.record(f"eval/reward_by_marker_area/{area_key}", reward_sum)
            if markers > 0:
                self.logger.record(
                    f"eval/reward_yield_per_marker/{area_key}",
                    reward_sum / markers,
                )

        for area_key, count in waste_area_counts.items():
            self.logger.record(f"eval/wasted_markers/{area_key}", count)
            total_markers = resolution_total_counts.get(area_key, 0)
            self.logger.record(f"eval/resolution_markers_total/{area_key}", total_markers)
            if total_markers > 0:
                self.logger.record(
                    f"eval/wasted_marker_rate/{area_key}",
                    count / total_markers,
                )
            actionable = max(0, int(total_markers) - int(count))
            opportunities = int(opportunity_area_counts.get(area_key, 0))
            avoidable_est = max(0, actionable - opportunities)
            self.logger.record(f"eval/waste_unavoidable/{area_key}", int(count))
            self.logger.record(f"eval/waste_actionable_marker_pool/{area_key}", actionable)
            self.logger.record(f"eval/waste_avoidable_est/{area_key}", avoidable_est)
            if actionable > 0:
                self.logger.record(
                    f"eval/waste_avoidable_rate_est/{area_key}",
                    avoidable_est / actionable,
                )
        self.logger.record("eval/wasted_markers_total", waste_total)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq == 0:
            self._run_eval()
        return True
