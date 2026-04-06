"""OpenSkill-based skill tracking for N-player checkpoint evaluation.

Drop-in replacement for EloTracker when --skill_tracking openskill is selected.
Public method signatures match EloTracker so OpponentPool, callbacks, and
PFSP sampling work without modification.

Persistence schema (openskill_state.json):
    {
        "type": "openskill",
        "ratings": {"checkpoint_id": {"mu": float, "sigma": float}, ...},
        "config": {"initial_mu": float, "initial_sigma": float}
    }
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

from openskill.models import PlackettLuce, PlackettLuceRating


class OpenSkillTracker:
    """Tracks skill ratings using OpenSkill's Plackett-Luce model.

    Public interface is duck-type compatible with EloTracker:
        register_checkpoint, get_rating, set_rating,
        update_ratings_two_player, update_ratings_multiplayer,
        expected_win_probability, get_leaderboard, remove_checkpoint.

    get_rating() returns mu.  CheckpointInfo.elo will store mu in this mode,
    so all pool sorting / pruning / tier logic remains valid.
    """

    def __init__(
        self,
        save_path: str,
        initial_mu: float = 1500.0,
        initial_sigma: float = 433.0,
        pl_tau: float=25.0,
        read_only: bool = False,
    ):
        self.model = PlackettLuce(tau=pl_tau)
        self._ratings: dict[str, PlackettLuceRating] = {}
        self._games_played: dict[str, int] = {}
        self.save_path = Path(save_path)
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.read_only = read_only
        self._last_mtime: float = 0.0

        if self.save_path and self.save_path.exists():
            self._load_state()

    # ------------------------------------------------------------------
    # Public interface (duck-type compatible with EloTracker)
    # ------------------------------------------------------------------

    def register_checkpoint(
        self,
        checkpoint_id: str,
        initial_elo: Optional[float] = None,
    ) -> None:
        """Register a new checkpoint.  initial_elo is interpreted as mu."""
        if checkpoint_id not in self._ratings:
            mu = initial_elo if initial_elo is not None else self.initial_mu
            self._ratings[checkpoint_id] = PlackettLuceRating(
                mu=mu, sigma=self.initial_sigma, name=checkpoint_id
            )
            self._games_played.setdefault(checkpoint_id, 0)
            self._save_state()

    def get_rating(self, checkpoint_id: str) -> float:
        """Return mu for the given checkpoint."""
        if checkpoint_id in self._ratings:
            return self._ratings[checkpoint_id].mu
        return self.initial_mu

    def get_sigma(self, checkpoint_id: str) -> float:
        """Return sigma for the given checkpoint."""
        if checkpoint_id in self._ratings:
            return self._ratings[checkpoint_id].sigma
        return self.initial_sigma

    def get_games_played(self, checkpoint_id: str) -> int:
        """Return number of games played for the given checkpoint."""
        return self._games_played.get(checkpoint_id, 0)

    def set_rating(self, checkpoint_id: str, elo: float) -> None:
        """Set mu, preserving existing sigma."""
        if checkpoint_id in self._ratings:
            old = self._ratings[checkpoint_id]
            self._ratings[checkpoint_id] = PlackettLuceRating(
                mu=elo, sigma=old.sigma, name=checkpoint_id
            )
        else:
            self._ratings[checkpoint_id] = PlackettLuceRating(
                mu=elo, sigma=self.initial_sigma, name=checkpoint_id
            )
        self._save_state()

    def update_ratings_multiplayer(
        self,
        player_ids: list[str],
        final_scores: list[int],
    ) -> dict[str, float]:
        """Update ratings after an N-player game via PlackettLuce.rate().

        Args:
            player_ids: Checkpoint IDs (any order).
            final_scores: Corresponding scores (higher is better).

        Returns:
            {checkpoint_id: new_mu} for every player.
        """
        n = len(player_ids)
        if n < 2:
            return {player_ids[0]: self.get_rating(player_ids[0])} if n == 1 else {}

        # Ensure all registered
        for pid in player_ids:
            if pid not in self._ratings:
                self.register_checkpoint(pid)

        # Compute 1-based ranks (1 = best).  Ties share the same rank.
        # Sort indices by descending score; Python sort is stable.
        indexed = sorted(enumerate(final_scores), key=lambda x: -x[1])
        ranks = [0] * n
        for rank_pos, (orig_idx, score) in enumerate(indexed):
            if rank_pos == 0:
                ranks[orig_idx] = 1
            else:
                prev_orig_idx = indexed[rank_pos - 1][0]
                if score == indexed[rank_pos - 1][1]:
                    ranks[orig_idx] = ranks[prev_orig_idx]  # tie
                else:
                    ranks[orig_idx] = rank_pos + 1

        # Build teams in player_ids order and call the model
        teams = [[self._ratings[pid]] for pid in player_ids]
        new_teams = self.model.rate(teams, ranks=ranks)

        # Apply updates (reconstruct with correct name for safety)
        new_mus: dict[str, float] = {}
        for i, pid in enumerate(player_ids):
            updated = new_teams[i][0]
            self._ratings[pid] = PlackettLuceRating(
                mu=updated.mu, sigma=updated.sigma, name=pid
            )
            new_mus[pid] = updated.mu
            self._games_played[pid] = self._games_played.get(pid, 0) + 1

        self._save_state()
        return new_mus

    def update_ratings_two_player(
        self,
        winner_id: str,
        loser_id: str,
        draw: bool = False,
    ) -> tuple[float, float]:
        """Two-player update delegated to update_ratings_multiplayer."""
        if draw:
            scores = [1, 1]   # same score → tied rank
        else:
            scores = [2, 1]   # winner higher
        result = self.update_ratings_multiplayer([winner_id, loser_id], scores)
        return result[winner_id], result[loser_id]

    def expected_win_probability(
        self,
        checkpoint_id: str,
        opponent_id: str,
    ) -> float:
        """Bradley-Terry pairwise win probability with uncertainty.

        P(i > j) = 1 / (1 + exp(-(mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2)))

        Returns 0.5 if either checkpoint is unregistered.
        """
        if checkpoint_id not in self._ratings or opponent_id not in self._ratings:
            return 0.5
        mu_i = self._ratings[checkpoint_id].mu
        mu_j = self._ratings[opponent_id].mu
        sigma_i = self._ratings[checkpoint_id].sigma
        sigma_j = self._ratings[opponent_id].sigma
        denom = math.sqrt(sigma_i ** 2 + sigma_j ** 2)
        if denom == 0.0:
            return 0.5
        return 1.0 / (1.0 + math.exp(-(mu_i - mu_j) / denom))

    def get_leaderboard(self, top_n: Optional[int] = None) -> list[tuple[str, float]]:
        """Return (checkpoint_id, mu) pairs sorted by mu descending."""
        board = sorted(self._ratings.items(), key=lambda x: -x[1].mu)
        result = [(pid, r.mu) for pid, r in board]
        if top_n is not None:
            return result[:top_n]
        return result

    def remove_checkpoint(self, checkpoint_id: str) -> None:
        """Remove a checkpoint from tracking."""
        self._ratings.pop(checkpoint_id, None)
        self._save_state()

    def recenter(self, target_mu: float = 1500.0) -> None:
        """Recenter all mus so the mean equals target_mu.

        This keeps the rating scale bounded without affecting relative ordering.
        """
        if not self._ratings:
            return
        mus = [r.mu for r in self._ratings.values()]
        mean_mu = sum(mus) / len(mus)
        shift = target_mu - mean_mu
        if shift == 0.0:
            return
        for pid, r in self._ratings.items():
            self._ratings[pid] = PlackettLuceRating(
                mu=r.mu + shift, sigma=r.sigma, name=pid
            )
        self._save_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        if self.save_path is None or self.read_only:
            return
        state = {
            "type": "openskill",
            "ratings": {
                pid: {"mu": r.mu, "sigma": r.sigma}
                for pid, r in self._ratings.items()
            },
            "games_played": dict(self._games_played),
            "config": {
                "initial_mu": self.initial_mu,
                "initial_sigma": self.initial_sigma,
            },
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.save_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.save_path)
        try:
            self._last_mtime = self.save_path.stat().st_mtime
        except OSError:
            pass

    def reload_state(self) -> None:
        """Reload ratings from disk (read-only use in subprocesses)."""
        if not (self.save_path and self.save_path.exists()):
            return
        try:
            mtime = self.save_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._last_mtime:
            return
        self._load_state()
        self._last_mtime = mtime

    def _load_state(self) -> None:
        try:
            with open(self.save_path, "r") as f:
                state = json.load(f)
            config = state.get("config", {})
            self.initial_mu = config.get("initial_mu", self.initial_mu)
            self.initial_sigma = config.get("initial_sigma", self.initial_sigma)
            for pid, data in state.get("ratings", {}).items():
                self._ratings[pid] = PlackettLuceRating(
                    mu=data["mu"], sigma=data["sigma"], name=pid
                )
            self._games_played = {
                pid: int(g) for pid, g in state.get("games_played", {}).items()
            }
        except Exception:
            pass  # start fresh on load failure
