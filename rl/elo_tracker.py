"""Elo rating system for tracking checkpoint performance.

This module provides an Elo-based rating system for evaluating and comparing
policy checkpoints in self-play training. It supports:
- Standard Elo rating updates after matches
- Multi-player Elo extensions
- Match history tracking
- Win rate computation per checkpoint pair
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .opponent_pool import CheckpointInfo


@dataclass
class MatchResult:
    """Records the result of a single match.

    Attributes:
        player_ids: List of checkpoint IDs that participated (in seat order).
        final_scores: Final scores for each player (same order as player_ids).
        winner_id: Checkpoint ID of the winner.
        timestamp: ISO timestamp when match was played.
        metadata: Additional match metadata (e.g., num_rounds, game_length).
    """
    player_ids: list[str]
    final_scores: list[int]
    winner_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        """Create from dictionary."""
        return cls(**data)

    @property
    def loser_ids(self) -> list[str]:
        """Get IDs of non-winning players."""
        return [pid for pid in self.player_ids if pid != self.winner_id]


@dataclass
class HeadToHeadStats:
    """Statistics for matches between two specific checkpoints.

    Attributes:
        checkpoint_a: First checkpoint ID.
        checkpoint_b: Second checkpoint ID.
        wins_a: Number of wins for checkpoint A.
        wins_b: Number of wins for checkpoint B.
        total_games: Total games played between these checkpoints.
    """
    checkpoint_a: str
    checkpoint_b: str
    wins_a: int = 0
    wins_b: int = 0
    total_games: int = 0

    @property
    def win_rate_a(self) -> float:
        """Win rate for checkpoint A (0.5 if no games played)."""
        if self.total_games == 0:
            return 0.5
        return self.wins_a / self.total_games

    @property
    def win_rate_b(self) -> float:
        """Win rate for checkpoint B (0.5 if no games played)."""
        if self.total_games == 0:
            return 0.5
        return self.wins_b / self.total_games

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HeadToHeadStats":
        """Create from dictionary."""
        return cls(**data)


class EloTracker:
    """Tracks Elo ratings for all checkpoints in the opponent pool.

    Implements standard Elo rating with optional extensions for multi-player
    games. Provides methods for updating ratings after matches and querying
    relative performance between checkpoints.

    Example:
        >>> tracker = EloTracker(save_path="elo_state.json")
        >>> tracker.register_checkpoint("ckpt_1000", initial_elo=1500.0)
        >>> tracker.register_checkpoint("ckpt_2000", initial_elo=1500.0)
        >>> tracker.update_ratings_two_player("ckpt_2000", "ckpt_1000")  # ckpt_2000 won
        >>> print(tracker.get_rating("ckpt_2000"))  # Higher than 1500
    """

    # Special ID for the "current" training policy (not yet checkpointed)
    CURRENT_POLICY_ID = "__current_policy__"

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_elo: float = 1500.0,
        save_path: Optional[str] = None,
    ):
        """Initialize the Elo tracker.

        Args:
            k_factor: K-factor for Elo calculations. Higher = more volatile.
            initial_elo: Starting Elo for new checkpoints.
            save_path: Optional path to save/load tracker state.
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.save_path = Path(save_path) if save_path else None

        # Checkpoint ID -> Elo rating
        self._ratings: dict[str, float] = {}

        # Match history (most recent first)
        self._match_history: list[MatchResult] = []

        # Head-to-head statistics: (id_a, id_b) -> HeadToHeadStats
        # Always stored with lexicographically smaller ID first
        self._head_to_head: dict[tuple[str, str], HeadToHeadStats] = {}

        # Load existing state if available
        if self.save_path and self.save_path.exists():
            self._load_state()

    def register_checkpoint(
        self,
        checkpoint_id: str,
        initial_elo: Optional[float] = None,
    ) -> None:
        """Register a new checkpoint with the tracker.

        Args:
            checkpoint_id: Unique identifier for the checkpoint.
            initial_elo: Starting Elo. Uses default if None.
        """
        if checkpoint_id not in self._ratings:
            self._ratings[checkpoint_id] = initial_elo or self.initial_elo
            self._save_state()

    def get_rating(self, checkpoint_id: str) -> float:
        """Get the current Elo rating for a checkpoint.

        Args:
            checkpoint_id: The checkpoint to query.

        Returns:
            Current Elo rating, or initial_elo if not registered.
        """
        return self._ratings.get(checkpoint_id, self.initial_elo)

    def set_rating(self, checkpoint_id: str, elo: float) -> None:
        """Manually set a checkpoint's Elo rating.

        Args:
            checkpoint_id: The checkpoint to update.
            elo: New Elo rating.
        """
        self._ratings[checkpoint_id] = elo
        self._save_state()

    def update_ratings_two_player(
        self,
        winner_id: str,
        loser_id: str,
        draw: bool = False,
    ) -> tuple[float, float]:
        """Update Elo ratings after a two-player match.

        Args:
            winner_id: ID of the winning checkpoint.
            loser_id: ID of the losing checkpoint.
            draw: If True, treat as a draw (0.5 score each).

        Returns:
            Tuple of (new_winner_elo, new_loser_elo).
        """
        # Ensure both are registered
        if winner_id not in self._ratings:
            self.register_checkpoint(winner_id)
        if loser_id not in self._ratings:
            self.register_checkpoint(loser_id)

        r_winner = self._ratings[winner_id]
        r_loser = self._ratings[loser_id]

        # Expected scores
        e_winner = self._expected_score(r_winner, r_loser)
        e_loser = 1.0 - e_winner

        # Actual scores
        if draw:
            s_winner = 0.5
            s_loser = 0.5
        else:
            s_winner = 1.0
            s_loser = 0.0

        # Update ratings
        new_winner_elo = r_winner + self.k_factor * (s_winner - e_winner)
        new_loser_elo = r_loser + self.k_factor * (s_loser - e_loser)

        self._ratings[winner_id] = new_winner_elo
        self._ratings[loser_id] = new_loser_elo

        # Update head-to-head stats
        self._update_head_to_head(winner_id, loser_id, winner_id if not draw else None)

        # Record match
        match = MatchResult(
            player_ids=[winner_id, loser_id],
            final_scores=[1, 0] if not draw else [0, 0],
            winner_id=winner_id if not draw else "",
            metadata={"draw": draw},
        )
        self._match_history.insert(0, match)

        self._save_state()

        return new_winner_elo, new_loser_elo

    def update_ratings_multiplayer(
        self,
        player_ids: list[str],
        final_scores: list[int],
    ) -> dict[str, float]:
        """Update Elo ratings after a multi-player match.

        Uses a modified Elo calculation where each player is compared
        pairwise against all others, with points distributed based on
        relative performance.

        Args:
            player_ids: List of checkpoint IDs in seat order.
            final_scores: Final scores for each player (same order).

        Returns:
            Dictionary mapping checkpoint ID to new Elo rating.
        """
        n = len(player_ids)
        if n < 2:
            return {player_ids[0]: self.get_rating(player_ids[0])} if n == 1 else {}

        # Ensure all registered
        for pid in player_ids:
            if pid not in self._ratings:
                self.register_checkpoint(pid)

        # Determine rankings (handle ties)
        sorted_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        rankings = [0] * n
        current_rank = 0
        for i, idx in enumerate(sorted_indices):
            if i > 0 and final_scores[sorted_indices[i]] < final_scores[sorted_indices[i-1]]:
                current_rank = i
            rankings[idx] = current_rank

        # Winner is the player with rank 0 (highest score)
        winner_idx = rankings.index(0)
        winner_id = player_ids[winner_idx]

        # Compute Elo changes using pairwise comparisons
        old_ratings = {pid: self._ratings[pid] for pid in player_ids}
        elo_changes = {pid: 0.0 for pid in player_ids}

        # Reduced K-factor for multiplayer (each pair comparison)
        k_mp = self.k_factor / (n - 1)

        for i in range(n):
            for j in range(i + 1, n):
                pid_i, pid_j = player_ids[i], player_ids[j]
                r_i, r_j = old_ratings[pid_i], old_ratings[pid_j]

                e_i = self._expected_score(r_i, r_j)
                e_j = 1.0 - e_i

                # Actual score based on ranking
                if rankings[i] < rankings[j]:
                    s_i, s_j = 1.0, 0.0
                elif rankings[i] > rankings[j]:
                    s_i, s_j = 0.0, 1.0
                else:
                    s_i, s_j = 0.5, 0.5

                elo_changes[pid_i] += k_mp * (s_i - e_i)
                elo_changes[pid_j] += k_mp * (s_j - e_j)

                # Update head-to-head
                h2h_winner = None
                if rankings[i] < rankings[j]:
                    h2h_winner = pid_i
                elif rankings[j] < rankings[i]:
                    h2h_winner = pid_j
                self._update_head_to_head(pid_i, pid_j, h2h_winner)

        # Apply changes
        new_ratings = {}
        for pid in player_ids:
            new_elo = old_ratings[pid] + elo_changes[pid]
            self._ratings[pid] = new_elo
            new_ratings[pid] = new_elo

        # Record match
        match = MatchResult(
            player_ids=player_ids,
            final_scores=final_scores,
            winner_id=winner_id,
            metadata={"multiplayer": True, "n_players": n},
        )
        self._match_history.insert(0, match)

        self._save_state()

        return new_ratings

    def get_head_to_head(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
    ) -> HeadToHeadStats:
        """Get head-to-head statistics between two checkpoints.

        Args:
            checkpoint_a: First checkpoint ID.
            checkpoint_b: Second checkpoint ID.

        Returns:
            HeadToHeadStats with win counts and rates.
        """
        key = self._h2h_key(checkpoint_a, checkpoint_b)
        if key not in self._head_to_head:
            return HeadToHeadStats(checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)

        stats = self._head_to_head[key]
        # Return with correct orientation
        if checkpoint_a == stats.checkpoint_a:
            return stats
        else:
            # Flip the stats
            return HeadToHeadStats(
                checkpoint_a=checkpoint_a,
                checkpoint_b=checkpoint_b,
                wins_a=stats.wins_b,
                wins_b=stats.wins_a,
                total_games=stats.total_games,
            )

    def get_win_rate(
        self,
        checkpoint_id: str,
        opponent_id: str,
    ) -> float:
        """Get win rate of checkpoint against a specific opponent.

        Args:
            checkpoint_id: The checkpoint to query.
            opponent_id: The opponent checkpoint.

        Returns:
            Win rate (0.0 to 1.0), or 0.5 if no games played.
        """
        stats = self.get_head_to_head(checkpoint_id, opponent_id)
        return stats.win_rate_a

    def get_leaderboard(self, top_n: Optional[int] = None) -> list[tuple[str, float]]:
        """Get checkpoints sorted by Elo rating (highest first).

        Args:
            top_n: If provided, return only top N entries.

        Returns:
            List of (checkpoint_id, elo) tuples sorted by rating.
        """
        sorted_ratings = sorted(self._ratings.items(), key=lambda x: -x[1])
        if top_n is not None:
            return sorted_ratings[:top_n]
        return sorted_ratings

    def get_recent_matches(self, n: int = 10) -> list[MatchResult]:
        """Get the N most recent matches.

        Args:
            n: Number of matches to return.

        Returns:
            List of MatchResult objects (most recent first).
        """
        return self._match_history[:n]

    def expected_win_probability(
        self,
        checkpoint_id: str,
        opponent_id: str,
    ) -> float:
        """Calculate expected win probability based on Elo ratings.

        Args:
            checkpoint_id: The checkpoint to query.
            opponent_id: The opponent checkpoint.

        Returns:
            Expected probability of checkpoint beating opponent.
        """
        r_self = self.get_rating(checkpoint_id)
        r_opp = self.get_rating(opponent_id)
        return self._expected_score(r_self, r_opp)

    def remove_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a checkpoint from tracking.

        Note: This removes the rating but preserves match history.

        Args:
            checkpoint_id: The checkpoint to remove.
        """
        self._ratings.pop(checkpoint_id, None)
        self._save_state()

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B.

        Uses the standard Elo formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _h2h_key(self, id_a: str, id_b: str) -> tuple[str, str]:
        """Get canonical key for head-to-head lookup."""
        return (min(id_a, id_b), max(id_a, id_b))

    def _update_head_to_head(
        self,
        id_a: str,
        id_b: str,
        winner_id: Optional[str],
    ) -> None:
        """Update head-to-head statistics after a match.

        Args:
            id_a: First player ID.
            id_b: Second player ID.
            winner_id: ID of winner, or None for draw.
        """
        key = self._h2h_key(id_a, id_b)

        if key not in self._head_to_head:
            self._head_to_head[key] = HeadToHeadStats(
                checkpoint_a=key[0],
                checkpoint_b=key[1],
            )

        stats = self._head_to_head[key]
        stats.total_games += 1

        if winner_id == stats.checkpoint_a:
            stats.wins_a += 1
        elif winner_id == stats.checkpoint_b:
            stats.wins_b += 1
        # Draw: neither wins_a nor wins_b incremented

    def _save_state(self) -> None:
        """Save tracker state to disk."""
        if self.save_path is None:
            return

        state = {
            "ratings": self._ratings,
            "match_history": [m.to_dict() for m in self._match_history[-1000:]],  # Keep last 1000
            "head_to_head": {
                f"{k[0]}|{k[1]}": v.to_dict()
                for k, v in self._head_to_head.items()
            },
            "config": {
                "k_factor": self.k_factor,
                "initial_elo": self.initial_elo,
            },
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load tracker state from disk."""
        if self.save_path is None or not self.save_path.exists():
            return

        try:
            with open(self.save_path, "r") as f:
                state = json.load(f)

            self._ratings = state.get("ratings", {})
            self._match_history = [
                MatchResult.from_dict(m) for m in state.get("match_history", [])
            ]
            self._head_to_head = {
                tuple(k.split("|")): HeadToHeadStats.from_dict(v)
                for k, v in state.get("head_to_head", {}).items()
            }

            # Update config if stored
            config = state.get("config", {})
            if "k_factor" in config:
                self.k_factor = config["k_factor"]
            if "initial_elo" in config:
                self.initial_elo = config["initial_elo"]

        except Exception:
            # If loading fails, start fresh
            pass
