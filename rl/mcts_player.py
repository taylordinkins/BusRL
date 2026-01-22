"""MCTS-enhanced player for Bus game.

This module provides a player wrapper that uses Monte Carlo Tree Search
with a trained policy network for action selection. Can be used for
evaluation, human vs AI play, or tournament matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from .bus_env import BusEnv

from .mcts import MCTS, MCTSConfig


@dataclass
class MCTSPlayerStats:
    """Statistics from MCTS player's decision making.

    Attributes:
        total_searches: Total number of MCTS searches performed.
        total_simulations: Total simulations across all searches.
        avg_root_value: Average root value estimate.
        avg_search_depth: Average search depth reached.
    """
    total_searches: int = 0
    total_simulations: int = 0
    avg_root_value: float = 0.0
    avg_search_depth: float = 0.0

    def update(self, stats: dict) -> None:
        """Update statistics with results from a search."""
        self.total_searches += 1
        self.total_simulations += stats.get("root_visits", 0)

        # Running average of root value
        new_value = stats.get("root_value", 0.0)
        self.avg_root_value = (
            (self.avg_root_value * (self.total_searches - 1) + new_value)
            / self.total_searches
        )


class MCTSPlayer:
    """Player that uses MCTS for action selection.

    Wraps a trained MaskablePPO model and enhances it with Monte Carlo
    Tree Search at inference time. The policy network provides action
    priors and value estimates for MCTS.

    Example:
        >>> model = MaskablePPO.load("bus_model.zip")
        >>> player = MCTSPlayer(model, n_simulations=100)
        >>> action = player.get_action(env)
    """

    def __init__(
        self,
        policy_model: "MaskablePPO",
        n_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 0.1,
        use_value_network: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,  # No noise by default for evaluation
        config: Optional[MCTSConfig] = None,
    ):
        """Initialize MCTS player.

        Args:
            policy_model: Trained MaskablePPO model.
            n_simulations: Number of MCTS simulations per move.
            c_puct: Exploration constant for PUCT formula.
            temperature: Temperature for action selection (0 = greedy).
            use_value_network: Use value network for leaf eval (vs rollout).
            dirichlet_alpha: Dirichlet noise alpha (for root exploration).
            dirichlet_epsilon: Weight of Dirichlet noise (0 = no noise).
            config: Full MCTSConfig (overrides individual params if provided).
        """
        self.model = policy_model

        if config is not None:
            self.config = config
        else:
            self.config = MCTSConfig(
                n_simulations=n_simulations,
                c_puct=c_puct,
                temperature=temperature,
                use_value_network=use_value_network,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )

        self.mcts = MCTS(policy_model, self.config)
        self.stats = MCTSPlayerStats()

    def get_action(self, env: "BusEnv") -> int:
        """Get action using MCTS search.

        Args:
            env: Current environment state.

        Returns:
            Best action index.
        """
        return self.mcts.search(env)

    def get_action_with_stats(self, env: "BusEnv") -> tuple[int, dict]:
        """Get action and search statistics.

        Args:
            env: Current environment state.

        Returns:
            Tuple of (action_index, stats_dict).
        """
        action, stats = self.mcts.search_with_stats(env)
        self.stats.update(stats)
        return action, stats

    def get_action_distribution(self, env: "BusEnv") -> tuple[int, np.ndarray]:
        """Get action and visit count distribution.

        Useful for analysis and training data generation.

        Args:
            env: Current environment state.

        Returns:
            Tuple of (action_index, visit_distribution).
        """
        action, stats = self.mcts.search_with_stats(env)
        self.stats.update(stats)

        # Convert visit counts to distribution
        total_actions = env.action_space.n
        visit_dist = np.zeros(total_actions, dtype=np.float32)

        child_visits = stats.get("child_visits", {})
        total_visits = sum(child_visits.values())

        if total_visits > 0:
            for a, v in child_visits.items():
                visit_dist[a] = v / total_visits

        return action, visit_dist

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self.stats = MCTSPlayerStats()

    def get_stats_summary(self) -> dict:
        """Get summary of player statistics.

        Returns:
            Dictionary with statistics summary.
        """
        return {
            "total_searches": self.stats.total_searches,
            "total_simulations": self.stats.total_simulations,
            "avg_simulations_per_search": (
                self.stats.total_simulations / self.stats.total_searches
                if self.stats.total_searches > 0
                else 0
            ),
            "avg_root_value": self.stats.avg_root_value,
            "config": {
                "n_simulations": self.config.n_simulations,
                "c_puct": self.config.c_puct,
                "temperature": self.config.temperature,
                "use_value_network": self.config.use_value_network,
            },
        }


class PolicyPlayer:
    """Player that uses policy network directly (no MCTS).

    Baseline player for comparison with MCTS player.
    """

    def __init__(
        self,
        policy_model: "MaskablePPO",
        deterministic: bool = True,
    ):
        """Initialize policy player.

        Args:
            policy_model: Trained MaskablePPO model.
            deterministic: Use deterministic action selection.
        """
        self.model = policy_model
        self.deterministic = deterministic

    def get_action(self, env: "BusEnv") -> int:
        """Get action from policy network.

        Args:
            env: Current environment state.

        Returns:
            Action index.
        """
        obs = env._get_observation()
        mask = env.action_masks()

        action, _ = self.model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )

        return int(action)


def compare_players(
    env: "BusEnv",
    mcts_player: MCTSPlayer,
    policy_player: PolicyPlayer,
    n_games: int = 10,
    verbose: bool = True,
) -> dict:
    """Compare MCTS player vs policy-only player.

    Plays games with each player and returns statistics.

    Args:
        env: Base environment to use.
        mcts_player: MCTS-enhanced player.
        policy_player: Policy-only player.
        n_games: Number of games to play.
        verbose: Print progress.

    Returns:
        Dictionary with comparison results.
    """
    mcts_wins = 0
    policy_wins = 0
    draws = 0

    mcts_scores = []
    policy_scores = []

    for game_idx in range(n_games):
        # Alternate who goes first
        mcts_slot = game_idx % 2
        policy_slot = 1 - mcts_slot

        game_env = env.clone()
        obs, info = game_env.reset()

        while not game_env._engine.is_game_over():
            current_player = game_env.get_current_player()

            if current_player == mcts_slot:
                action = mcts_player.get_action(game_env)
            else:
                action = policy_player.get_action(game_env)

            obs, reward, terminated, truncated, info = game_env.step(action)

            if terminated or truncated:
                break

        # Get final scores
        scores = info.get("scores", {})
        mcts_score = scores.get(mcts_slot, 0)
        policy_score = scores.get(policy_slot, 0)

        mcts_scores.append(mcts_score)
        policy_scores.append(policy_score)

        if mcts_score > policy_score:
            mcts_wins += 1
        elif policy_score > mcts_score:
            policy_wins += 1
        else:
            draws += 1

        if verbose:
            print(f"Game {game_idx + 1}: MCTS {mcts_score} - Policy {policy_score}")

    results = {
        "n_games": n_games,
        "mcts_wins": mcts_wins,
        "policy_wins": policy_wins,
        "draws": draws,
        "mcts_win_rate": mcts_wins / n_games,
        "policy_win_rate": policy_wins / n_games,
        "mcts_avg_score": np.mean(mcts_scores),
        "policy_avg_score": np.mean(policy_scores),
        "mcts_stats": mcts_player.get_stats_summary(),
    }

    if verbose:
        print(f"\nResults: MCTS {mcts_wins}W / Policy {policy_wins}W / {draws}D")
        print(f"MCTS win rate: {results['mcts_win_rate']:.1%}")

    return results
