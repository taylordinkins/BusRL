"""Multi-policy environment wrapper for diverse opponent training.

This module provides an environment wrapper that supports different policies
for different player slots, enabling training against a pool of past checkpoints
rather than always playing against the current policy (pure self-play).

Key features:
- Assign different policies to different player slots
- Training policy only collects experiences on its designated slot
- Opponents are sampled from the checkpoint pool
- Automatic opponent action selection during non-training turns
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any, Callable, TYPE_CHECKING
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from .opponent_pool import OpponentPool, CheckpointInfo
    from .elo_tracker import EloTracker


class PolicySlot:
    """Represents a policy assigned to a player slot.

    Attributes:
        policy: The loaded policy model (MaskablePPO or similar).
        checkpoint_id: ID of the checkpoint this policy came from.
        is_training_policy: True if this is the policy being trained.
    """

    def __init__(
        self,
        policy: Optional["MaskablePPO"],
        checkpoint_id: str,
        is_training_policy: bool = False,
    ):
        self.policy = policy
        self.checkpoint_id = checkpoint_id
        self.is_training_policy = is_training_policy

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Get action from this policy.

        Args:
            obs: Observation array.
            action_mask: Boolean mask of valid actions.

        Returns:
            Selected action index.
        """
        if self.policy is None:
            # Random valid action fallback
            import warnings
            warnings.warn(f"PolicySlot {self.checkpoint_id} has None policy, using random actions")
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                return 0
            return np.random.choice(valid_actions)

        # Use the policy to predict action
        action, _ = self.policy.predict(obs, deterministic=False, action_masks=action_mask)
        return int(action)


class MultiPolicyBusEnv(gym.Wrapper):
    """Environment wrapper supporting different policies per player slot.

    This wrapper enables training against a diverse set of opponents by:
    1. Assigning the training policy to one player slot
    2. Sampling opponents from a checkpoint pool for other slots
    3. Automatically executing opponent turns
    4. Only returning observations/rewards for the training policy's turns

    The wrapper is transparent to the training algorithm - from its perspective,
    it's still a single-agent environment, just one where opponents vary.

    Example:
        >>> from rl.opponent_pool import OpponentPool
        >>> pool = OpponentPool(save_dir="checkpoints")
        >>> env = BusEnv(num_players=4)
        >>> multi_env = MultiPolicyBusEnv(
        ...     env,
        ...     opponent_pool=pool,
        ...     training_slot=0,
        ...     self_play_prob=0.2,
        ... )
        >>> obs, info = multi_env.reset()
        >>> # Now step() will auto-handle opponent turns
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_pool: Optional["OpponentPool"] = None,
        training_slot: int = 0,
        self_play_prob: float = 0.2,
        sampling_method: str = "pfsp",
        elo_tracker: Optional["EloTracker"] = None,
        randomize_training_slot: bool = False,
        self_play_checkpoint_path: Optional[str] = None,
    ):
        """Initialize the multi-policy wrapper.

        Args:
            env: The base BusEnv environment.
            opponent_pool: Pool to sample opponents from. If None, pure self-play.
            training_slot: Player slot for the training policy (0 = first player).
                If randomize_training_slot is True, this is used as the initial slot.
            self_play_prob: Probability of using current policy as opponent.
            sampling_method: How to sample from pool ("uniform", "pfsp", "elo_weighted").
            elo_tracker: Optional Elo tracker for rating updates after games.
            randomize_training_slot: If True, randomize training slot each episode.
                This helps the agent learn to play from any position.
            self_play_checkpoint_path: Path to checkpoint for self-play opponents.
                Required for SubprocVecEnv compatibility when self_play_prob > 0.
                Each subprocess loads this checkpoint independently.
        """
        super().__init__(env)

        self.opponent_pool = opponent_pool
        self._initial_training_slot = training_slot
        self.training_slot = training_slot
        self.self_play_prob = self_play_prob
        self.sampling_method = sampling_method
        self.elo_tracker = elo_tracker
        self.randomize_training_slot = randomize_training_slot
        self.self_play_checkpoint_path = self_play_checkpoint_path

        # Policy assignments for each player slot
        self._policy_slots: list[PolicySlot] = []
        self._num_players = getattr(env, "num_players", 4)

        # Cached training policy reference (set externally for DummyVecEnv)
        # For SubprocVecEnv, use self_play_checkpoint_path instead
        self._training_policy: Optional["MaskablePPO"] = None

        # Cached self-play policy loaded from checkpoint (for SubprocVecEnv)
        self._self_play_checkpoint_policy: Optional["MaskablePPO"] = None

        # Track game outcomes for Elo updates
        self._current_episode_checkpoints: list[str] = []

    @property
    def training_policy(self) -> Optional["MaskablePPO"]:
        """Get the current training policy."""
        return self._training_policy

    @training_policy.setter
    def training_policy(self, policy: "MaskablePPO") -> None:
        """Set the current training policy."""
        self._training_policy = policy
        # Also update pool's current policy reference
        if self.opponent_pool is not None:
            self.opponent_pool.current_policy = policy

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment and assign opponents for this episode.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info) from training player's perspective.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # For subprocess environments, refresh the opponent pool from disk
        # to pick up new checkpoints saved by the main training process.
        # We detect subprocess mode by checking for self_play_checkpoint_path
        # (only set in SubprocVecEnv mode) or elo_tracker being None.
        if self.opponent_pool is not None and self.elo_tracker is None:
            self.opponent_pool.refresh()

        # Randomize training slot if enabled (learn to play from any position)
        if self.randomize_training_slot:
            self.training_slot = np.random.randint(0, self._num_players)
        else:
            self.training_slot = self._initial_training_slot

        # Assign policies to player slots
        self._assign_policies()

        # Store checkpoint IDs for this episode (for Elo updates)
        self._current_episode_checkpoints = [
            slot.checkpoint_id for slot in self._policy_slots
        ]

        # If not training player's turn, advance to their turn
        obs, info = self._advance_to_training_turn(obs, info)

        return obs, info

    def step(
        self,
        action: Union[int, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute training policy's action and handle opponent turns.

        Args:
            action: Action chosen by the training policy.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Execute training policy's action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track acting player for reward attribution
        acting_player = self.training_slot
        cumulative_reward = float(reward)

        # Handle opponent turns until game ends or training player's turn
        if not (terminated or truncated):
            terminated_before_opp = False
            obs, info, term_during_opp, trunc_during_opp, opp_rewards = self._handle_opponent_turns(obs, info)
            
            # Update terminal/truncated flags
            terminated = terminated or term_during_opp
            truncated = truncated or trunc_during_opp

            # If the game ended during opponent turns, the training player
            # needs to receive its terminal reward (win/loss differential).
            # BusEnv.step only computes rewards for the current acting player.
            if terminated and not terminated_before_opp:
                try:
                    reward_calculator = self.env.get_wrapper_attr("_reward_calculator")
                    terminal_reward = reward_calculator.compute_reward(
                        state=self.env.unwrapped._engine.state,
                        prev_state=self.env.unwrapped._prev_state, # Use prev state from engine
                        player_id=self.training_slot,
                        done=True
                    )
                    cumulative_reward += float(terminal_reward)
                except (AttributeError, Exception):
                    # Fallback if reward calculator is not accessible
                    pass
        
        # Game ended - update Elo ratings if tracker available
        if terminated and self.elo_tracker is not None:
            self._update_elo_ratings(info)

        info["training_slot"] = self.training_slot
        info["acting_player"] = acting_player
        info["opponent_checkpoints"] = self._current_episode_checkpoints
        info["cumulative_reward"] = cumulative_reward

        return obs, cumulative_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Get action mask for the training policy.

        Returns:
            Boolean array of valid actions.
        """
        try:
            return self.env.action_masks()
        except AttributeError:
            return self.env.get_wrapper_attr("action_masks")()

    def _assign_policies(self) -> None:
        """Assign policies to all player slots for this episode."""
        self._policy_slots = []

        for slot in range(self._num_players):
            if slot == self.training_slot:
                # Training slot uses current training policy
                # In SubprocVecEnv, _training_policy is None, so we use self-play checkpoint
                if self._training_policy is None:
                    # Subprocess mode: load from self_play_checkpoint_path
                    frozen_policy = self._get_self_play_policy()
                    self._policy_slots.append(PolicySlot(
                        policy=frozen_policy,
                        checkpoint_id="__training_subprocess__",
                        is_training_policy=True,
                    ))
                else:
                    # Main process mode: use live training policy reference
                    self._policy_slots.append(PolicySlot(
                        policy=self._training_policy,
                        checkpoint_id="__current__",
                        is_training_policy=True,
                    ))
            else:
                # Sample opponent for this slot
                policy_slot = self._sample_opponent_policy()
                self._policy_slots.append(policy_slot)

    def _create_frozen_copy(self, policy: "MaskablePPO") -> "MaskablePPO":
        """Create a frozen copy of a policy for opponent use.

        This ensures opponents use a snapshot of the current weights rather than
        a live reference that changes as training progresses. This prevents
        co-evolution where opponents and training policy evolve in lockstep.

        Args:
            policy: The policy to copy.

        Returns:
            A frozen copy of the policy with gradients disabled.
        """
        # Using deepcopy on PyTorch models can cause issues in multiprocessing
        # Instead, save to a temporary file and reload
        import tempfile
        import os
        from sb3_contrib import MaskablePPO

        # Create a temporary file for the checkpoint
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.zip') as tmp:
            tmp_path = tmp.name

        try:
            # Save the policy
            policy.save(tmp_path)
            
            # Load it back (creates independent copy)
            frozen = MaskablePPO.load(tmp_path)
            
            # Put in evaluation mode and disable gradients
            frozen.policy.set_training_mode(False)
            for param in frozen.policy.parameters():
                param.requires_grad = False
            
            return frozen
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass

    def _get_self_play_policy(self) -> "MaskablePPO":
        """Get a frozen policy for self-play opponents.

        This method handles both DummyVecEnv (uses training_policy reference)
        and SubprocVecEnv (loads from checkpoint path) scenarios.

        Returns:
            Frozen policy for self-play opponent, or None if unavailable.
        """
        # Option 1: Use training_policy reference (DummyVecEnv)
        # Creates a fresh frozen copy each time to capture current weights
        if self._training_policy is not None:
            return self._create_frozen_copy(self._training_policy)

        # Option 2: Load from checkpoint path (SubprocVecEnv)
        # Reload each time to pick up updates from OpponentPoolCallback
        if self.self_play_checkpoint_path is not None:
            from sb3_contrib import MaskablePPO
            import os

            # Check both with and without .zip extension
            path = self.self_play_checkpoint_path
            if not path.endswith(".zip"):
                path_with_ext = path + ".zip"
            else:
                path_with_ext = path

            if not os.path.exists(path_with_ext):
                # Checkpoint doesn't exist yet, return None
                return None

            try:
                loaded = MaskablePPO.load(self.self_play_checkpoint_path)
                # Freeze it
                loaded.policy.set_training_mode(False)
                for param in loaded.policy.parameters():
                    param.requires_grad = False
                return loaded
            except Exception:
                # Loading failed, return None
                return None

        # Option 3: No policy available - return None (will use random actions)
        return None

    def _sample_opponent_policy(self) -> PolicySlot:
        """Sample an opponent policy from the pool.

        Returns:
            PolicySlot with loaded opponent policy.
        """
        # Self-play: use frozen policy (either from training_policy or checkpoint)
        # This prevents co-evolution where the opponent's behavior changes
        # as the training policy updates mid-episode
        if self.opponent_pool is None or np.random.random() < self.self_play_prob:
            frozen_policy = self._get_self_play_policy()
            return PolicySlot(
                policy=frozen_policy,
                checkpoint_id="__current_snapshot__",
                is_training_policy=False,
            )

        # Sample from pool
        if len(self.opponent_pool) == 0:
            # Pool empty, fall back to frozen self-play
            frozen_policy = self._get_self_play_policy()
            return PolicySlot(
                policy=frozen_policy,
                checkpoint_id="__current_snapshot__",
                is_training_policy=False,
            )

        checkpoint_info = self.opponent_pool.sample_opponent(method=self.sampling_method)
        if checkpoint_info is None:
            frozen_policy = self._get_self_play_policy()
            return PolicySlot(
                policy=frozen_policy,
                checkpoint_id="__current_snapshot__",
                is_training_policy=False,
            )

        # Load the checkpoint (already frozen by opponent_pool.load_checkpoint)
        try:
            loaded_policy = self.opponent_pool.load_checkpoint(checkpoint_info)
            return PolicySlot(
                policy=loaded_policy,
                checkpoint_id=checkpoint_info.checkpoint_id,
                is_training_policy=False,
            )
        except Exception:
            # Loading failed, fall back to frozen self-play
            frozen_policy = self._get_self_play_policy()
            return PolicySlot(
                policy=frozen_policy,
                checkpoint_id="__current_snapshot__",
                is_training_policy=False,
            )

    def _advance_to_training_turn(
        self,
        obs: np.ndarray,
        info: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Advance through opponent turns until training player's turn.

        Args:
            obs: Current observation.
            info: Current info dict.

        Returns:
            Updated (observation, info) when it's training player's turn.
        """
        obs, info, _, _, _ = self._handle_opponent_turns(obs, info)
        return obs, info

    def _handle_opponent_turns(
        self,
        obs: np.ndarray,
        info: dict,
    ) -> Tuple[np.ndarray, dict, bool, bool, float]:
        """Execute opponent turns until training player's turn or game end.

        Args:
            obs: Current observation.
            info: Current info dict.

        Returns:
            Tuple of (obs, info, terminated, truncated, cumulative_opponent_rewards).
        """
        terminated = False
        truncated = False
        total_opponent_reward = 0.0

        max_iterations = 500  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Check current player
            try:
                current_player = self.env.get_current_player()
            except AttributeError:
                current_player = self.env.get_wrapper_attr("get_current_player")()

            # If training player's turn, we're done
            if current_player == self.training_slot:
                break

            # Check if game ended
            if self.env.unwrapped._engine.is_game_over():
                terminated = True
                break

            # Get opponent's action
            policy_slot = self._policy_slots[current_player]
            action_mask = self.env.action_masks()

            # Check if any valid actions
            if not np.any(action_mask):
                # No valid actions, game might be stuck or transitioning
                break

            action = policy_slot.get_action(obs, action_mask)

            # Execute opponent's action
            obs, opp_reward, terminated, truncated, info = self.env.step(action)
            total_opponent_reward += opp_reward

            if terminated or truncated:
                break

        return obs, info, terminated, truncated, total_opponent_reward

    def _update_elo_ratings(self, info: dict) -> None:
        """Update Elo ratings after a game ends.

        Args:
            info: Info dict from final step (should contain scores).
        """
        if self.elo_tracker is None:
            return

        scores = info.get("scores", {})
        if not scores:
            return

        # Build player_ids and final_scores lists
        player_ids = self._current_episode_checkpoints
        final_scores = [scores.get(i, 0) for i in range(len(player_ids))]

        # Update ratings
        self.elo_tracker.update_ratings_multiplayer(player_ids, final_scores)

        # Also update win rates in opponent pool
        if self.opponent_pool is not None:
            winner_idx = final_scores.index(max(final_scores))
            winner_id = player_ids[winner_idx]

            for checkpoint_id in player_ids:
                if checkpoint_id == "__current__":
                    continue

                checkpoint = self.opponent_pool.get_checkpoint_by_id(checkpoint_id)
                if checkpoint is None:
                    continue

                # Update win rate against training policy
                h2h = self.elo_tracker.get_head_to_head(checkpoint_id, "__current__")
                if h2h.total_games > 0:
                    # Win rate FROM checkpoint's perspective against training
                    win_rate = h2h.win_rate_a if h2h.checkpoint_a == checkpoint_id else h2h.win_rate_b
                    self.opponent_pool.update_checkpoint_stats(
                        checkpoint_id=checkpoint_id,
                        win_rate=win_rate,
                        elo=self.elo_tracker.get_rating(checkpoint_id),
                        games_played_delta=1,
                    )


class MatchRunner:
    """Utility class for running evaluation matches between checkpoints.

    Used by the EvalCallback to play games between the training policy
    and pool checkpoints to update win rates and Elo ratings.

    Example:
        >>> runner = MatchRunner(env_factory, elo_tracker)
        >>> results = runner.run_match(policy_a, policy_b, n_games=10)
        >>> print(f"Policy A win rate: {results['win_rate_a']}")
    """

    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        elo_tracker: Optional["EloTracker"] = None,
    ):
        """Initialize the match runner.

        Args:
            env_factory: Function that creates a fresh BusEnv.
            elo_tracker: Optional Elo tracker for rating updates.
        """
        self.env_factory = env_factory
        self.elo_tracker = elo_tracker

    def run_match(
        self,
        policy_a: "MaskablePPO",
        policy_b: "MaskablePPO",
        checkpoint_id_a: str,
        checkpoint_id_b: str,
        n_games: int = 10,
        randomize_seats: bool = True,
    ) -> dict:
        """Run multiple games between two policies.

        Args:
            policy_a: First policy.
            policy_b: Second policy.
            checkpoint_id_a: ID for first policy.
            checkpoint_id_b: ID for second policy.
            n_games: Number of games to play.
            randomize_seats: If True, alternate which policy goes first.

        Returns:
            Dictionary with match statistics.
        """
        wins_a = 0
        wins_b = 0
        draws = 0
        total_scores_a = 0
        total_scores_b = 0

        for game_idx in range(n_games):
            # Alternate seats if randomizing
            if randomize_seats and game_idx % 2 == 1:
                policies = [policy_b, policy_a]
                ids = [checkpoint_id_b, checkpoint_id_a]
            else:
                policies = [policy_a, policy_b]
                ids = [checkpoint_id_a, checkpoint_id_b]

            result = self._run_single_game(policies, ids)

            # Attribute to original a/b
            if randomize_seats and game_idx % 2 == 1:
                # Seats were swapped
                score_a = result["scores"][1]
                score_b = result["scores"][0]
            else:
                score_a = result["scores"][0]
                score_b = result["scores"][1]

            total_scores_a += score_a
            total_scores_b += score_b

            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
            else:
                draws += 1

            # Update Elo after each game
            if self.elo_tracker is not None:
                if score_a > score_b:
                    self.elo_tracker.update_ratings_two_player(checkpoint_id_a, checkpoint_id_b)
                elif score_b > score_a:
                    self.elo_tracker.update_ratings_two_player(checkpoint_id_b, checkpoint_id_a)
                else:
                    self.elo_tracker.update_ratings_two_player(checkpoint_id_a, checkpoint_id_b, draw=True)

        return {
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "total_games": n_games,
            "win_rate_a": wins_a / n_games if n_games > 0 else 0.5,
            "win_rate_b": wins_b / n_games if n_games > 0 else 0.5,
            "avg_score_a": total_scores_a / n_games if n_games > 0 else 0,
            "avg_score_b": total_scores_b / n_games if n_games > 0 else 0,
        }

    def _run_single_game(
        self,
        policies: list["MaskablePPO"],
        checkpoint_ids: list[str],
    ) -> dict:
        """Run a single game with given policies.

        Args:
            policies: List of policies for each player slot (typically 2 for head-to-head).
            checkpoint_ids: IDs for each policy.

        Returns:
            Game result with final scores.

        Note:
            If the environment requires more than 2 players, extra slots will be
            filled with copies of policy_b to make it policy_a vs N copies of policy_b.
        """
        env = self.env_factory()

        # Get the required number of players from the environment
        num_players = getattr(env, "num_players", len(policies))

        # If we have exactly the right number of policies, use them directly
        if len(policies) == num_players:
            expanded_policies = policies
            expanded_checkpoint_ids = checkpoint_ids
        # If we have 2 policies but need more players, fill extra slots with policy_b
        elif len(policies) == 2 and num_players > 2:
            policy_a, policy_b = policies
            checkpoint_id_a, checkpoint_id_b = checkpoint_ids
            # Put policy_a in slot 0, fill rest with policy_b
            expanded_policies = [policy_a] + [policy_b] * (num_players - 1)
            expanded_checkpoint_ids = [checkpoint_id_a] + [f"{checkpoint_id_b}_slot{i}" for i in range(1, num_players)]
        else:
            raise ValueError(
                f"Cannot run game: env requires {num_players} players but got {len(policies)} policies. "
                f"Expected either {num_players} or 2 policies."
            )

        obs, _ = env.reset()
        terminated = truncated = False

        max_steps = 2000
        step = 0

        while not (terminated or truncated) and step < max_steps:
            step += 1

            current_player = env.get_current_player()
            action_mask = env.action_masks()

            # Get action from current player's policy
            policy = expanded_policies[current_player]
            action, _ = policy.predict(obs, deterministic=True, action_masks=action_mask)

            obs, _, terminated, truncated, info = env.step(int(action))

        # Get final scores
        final_info = env._get_info() if hasattr(env, "_get_info") else info
        scores = final_info.get("scores", {i: 0 for i in range(num_players)})

        env.close()

        # Return only the scores for the original 2 policies (slots 0 and 1)
        return {
            "scores": [scores.get(0, 0), scores.get(1, 0)],
            "checkpoint_ids": checkpoint_ids,
        }

    def evaluate_against_pool(
        self,
        current_policy: "MaskablePPO",
        current_id: str,
        opponent_pool: "OpponentPool",
        n_games_per_opponent: int = 5,
        max_opponents: int = 10,
    ) -> dict:
        """Evaluate current policy against multiple pool checkpoints.

        Args:
            current_policy: The policy being evaluated.
            current_id: ID for the current policy.
            opponent_pool: Pool to sample opponents from.
            n_games_per_opponent: Games to play against each opponent.
            max_opponents: Maximum opponents to evaluate against.

        Returns:
            Aggregated evaluation results.
        """
        if len(opponent_pool) == 0:
            return {"error": "Empty opponent pool"}

        # Sample opponents (prefer diverse Elo range)
        opponents = opponent_pool.sample_opponents(
            n=min(max_opponents, len(opponent_pool)),
            method="uniform",
            allow_duplicates=False,
        )

        total_wins = 0
        total_games = 0
        results_by_opponent = {}

        for opponent_info in opponents:
            try:
                opponent_policy = opponent_pool.load_checkpoint(opponent_info)
                match_result = self.run_match(
                    policy_a=current_policy,
                    policy_b=opponent_policy,
                    checkpoint_id_a=current_id,
                    checkpoint_id_b=opponent_info.checkpoint_id,
                    n_games=n_games_per_opponent,
                )

                total_wins += match_result["wins_a"]
                total_games += match_result["total_games"]

                results_by_opponent[opponent_info.checkpoint_id] = {
                    "win_rate": match_result["win_rate_a"],
                    "opponent_elo": opponent_info.elo,
                }

                # Update opponent's stats in pool
                opponent_pool.update_checkpoint_stats(
                    checkpoint_id=opponent_info.checkpoint_id,
                    win_rate=match_result["win_rate_b"],  # From opponent's perspective
                    games_played_delta=n_games_per_opponent,
                )

            except Exception as e:
                # Skip failed opponent loads
                continue

        return {
            "total_wins": total_wins,
            "total_games": total_games,
            "overall_win_rate": total_wins / total_games if total_games > 0 else 0.5,
            "opponents_evaluated": len(results_by_opponent),
            "results_by_opponent": results_by_opponent,
        }
