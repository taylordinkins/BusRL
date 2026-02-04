"""Opponent Pool for self-play training.

This module provides checkpoint management and opponent sampling for
diverse self-play training. It supports:
- Saving and loading policy checkpoints
- Uniform and weighted sampling strategies
- Pool pruning to maintain diversity
- PFSP matchmaking with Elo-based opponent selection
- Integration with EloTracker for rating updates
"""

from __future__ import annotations

import os
import json
import random
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from .elo_tracker import EloTracker


@dataclass
class CheckpointInfo:
    """Metadata for a saved policy checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        path: Filesystem path to the saved model.
        step: Training step when checkpoint was saved.
        elo: Elo rating of this checkpoint (default 1500).
        created_at: ISO timestamp when checkpoint was created.
        win_rate_vs_current: Win rate against the current training policy.
        games_played: Number of evaluation games played.
        metadata: Additional custom metadata.
    """
    checkpoint_id: str
    path: str
    step: int
    elo: float = 1500.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    win_rate_vs_current: float = 0.5
    games_played: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PoolConfig:
    """Configuration for the opponent pool.

    Attributes:
        pool_size: Maximum number of checkpoints to retain.
        save_interval: Training steps between checkpoint saves.
        min_elo_gap: Minimum Elo difference to keep similar checkpoints.
        initial_elo: Starting Elo for new checkpoints.
        prune_strategy: Strategy for removing checkpoints ("oldest", "lowest_elo", "least_diverse").
    """
    pool_size: int = 20
    save_interval: int = 50_000
    min_elo_gap: int = 50
    initial_elo: float = 1500.0
    prune_strategy: str = "oldest"


class OpponentPool:
    """Manages a pool of past policy checkpoints for diverse training.

    The opponent pool stores frozen copies of the training policy at
    regular intervals. During training, opponents are sampled from this
    pool to provide diverse training signal and prevent policy collapse.

    Example:
        >>> pool = OpponentPool(save_dir="checkpoints", config=PoolConfig(pool_size=20))
        >>> pool.save_checkpoint(model, step=50000)
        >>> opponent = pool.sample_opponent(method="uniform")
    """

    def __init__(
        self,
        save_dir: str,
        config: Optional[PoolConfig] = None,
        elo_tracker: Optional["EloTracker"] = None,
    ):
        """Initialize the opponent pool.

        Args:
            save_dir: Directory to store checkpoint files.
            config: Pool configuration. Uses defaults if None.
            elo_tracker: Optional EloTracker for integrated rating management.
        """
        self.save_dir = Path(save_dir)
        self.config = config or PoolConfig()
        self.checkpoints: list[CheckpointInfo] = []
        self._current_policy: Optional[MaskablePPO] = None
        self._elo_tracker: Optional["EloTracker"] = elo_tracker

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoints if any
        self._load_pool_state()

    @property
    def elo_tracker(self) -> Optional["EloTracker"]:
        """Get the Elo tracker instance."""
        return self._elo_tracker

    @elo_tracker.setter
    def elo_tracker(self, tracker: "EloTracker") -> None:
        """Set the Elo tracker and sync existing checkpoints."""
        self._elo_tracker = tracker
        # Register all existing checkpoints with the tracker
        if tracker is not None:
            for checkpoint in self.checkpoints:
                tracker.register_checkpoint(checkpoint.checkpoint_id, checkpoint.elo)

    @property
    def current_policy(self) -> Optional["MaskablePPO"]:
        """Get the current training policy."""
        return self._current_policy

    @current_policy.setter
    def current_policy(self, policy: "MaskablePPO") -> None:
        """Set the current training policy."""
        self._current_policy = policy

    def save_checkpoint(
        self,
        model: "MaskablePPO",
        step: int,
        elo: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> CheckpointInfo:
        """Save the current policy as a new checkpoint.

        Args:
            model: The MaskablePPO model to save.
            step: Current training step.
            elo: Optional Elo rating. Uses initial_elo if None.
            metadata: Optional additional metadata to store.

        Returns:
            CheckpointInfo for the saved checkpoint.
        """
        # Generate unique checkpoint ID
        checkpoint_id = f"ckpt_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.save_dir / checkpoint_id

        # Save the model
        model.save(str(checkpoint_path))

        # Create checkpoint info
        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            path=str(checkpoint_path) + ".zip",  # SB3 adds .zip extension
            step=step,
            elo=elo if elo is not None else self.config.initial_elo,
            metadata=metadata or {},
        )

        self.checkpoints.append(info)

        # Register with Elo tracker if available
        if self._elo_tracker is not None:
            self._elo_tracker.register_checkpoint(checkpoint_id, info.elo)

        # Prune if over capacity
        if len(self.checkpoints) > self.config.pool_size:
            self._prune_pool()

        # Save pool state
        self._save_pool_state()

        return info

    def load_checkpoint(self, checkpoint_info: CheckpointInfo) -> "MaskablePPO":
        """Load a checkpoint from disk as a frozen policy.

        The loaded checkpoint is explicitly frozen (eval mode, no gradients)
        to ensure it's only used for inference. This prevents any accidental
        training of opponent policies and ensures consistent behavior from
        dropout/batch norm layers.

        Args:
            checkpoint_info: The checkpoint to load.

        Returns:
            Loaded and frozen MaskablePPO model.
        """
        from sb3_contrib import MaskablePPO

        # Handle path with or without .zip extension
        path = checkpoint_info.path
        if not path.endswith(".zip"):
            path = path + ".zip"

        loaded = MaskablePPO.load(path, device="cpu")

        # Explicitly freeze for inference only
        # This disables dropout, batch norm updates, etc.
        loaded.policy.set_training_mode(False)

        # Disable gradients on all parameters to prevent any accidental training
        for param in loaded.policy.parameters():
            param.requires_grad = False

        return loaded

    def sample_opponent(
        self,
        method: str = "uniform",
        exclude_ids: Optional[set[str]] = None,
    ) -> Optional[CheckpointInfo]:
        """Sample an opponent from the pool.

        Args:
            method: Sampling method ("uniform", "latest", "pfsp", "elo_weighted").
            exclude_ids: Checkpoint IDs to exclude from sampling.

        Returns:
            CheckpointInfo for the sampled opponent, or None if pool is empty.
        """
        if not self.checkpoints:
            return None

        candidates = self.checkpoints
        if exclude_ids:
            candidates = [c for c in candidates if c.checkpoint_id not in exclude_ids]

        if not candidates:
            return None

        if method == "uniform":
            return random.choice(candidates)

        elif method == "latest":
            # Weight towards more recent checkpoints
            weights = [i + 1 for i in range(len(candidates))]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(candidates, weights=probs, k=1)[0]

        elif method == "pfsp":
            # Prioritized Fictitious Self-Play: weight by uncertainty.
            # Use Elo-derived expected win probability when available — this
            # updates implicitly for all opponents whenever __current__'s Elo
            # changes, avoiding staleness of per-opponent win_rate fields.
            if self._elo_tracker is not None:
                weights = [self._pfsp_weight(
                    self._elo_tracker.expected_win_probability("__current__", c.checkpoint_id)
                ) for c in candidates]
            else:
                weights = [self._pfsp_weight(c.win_rate_vs_current) for c in candidates]
            total = sum(weights)
            if total == 0:
                return random.choice(candidates)
            probs = [w / total for w in weights]
            return random.choices(candidates, weights=probs, k=1)[0]

        elif method == "elo_weighted":
            # Sample proportionally to Elo (higher Elo = more likely)
            min_elo = min(c.elo for c in candidates)
            weights = [c.elo - min_elo + 100 for c in candidates]  # Shift to avoid zero
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(candidates, weights=probs, k=1)[0]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def sample_opponents(
        self,
        n: int,
        method: str = "uniform",
        allow_duplicates: bool = True,
    ) -> list[CheckpointInfo]:
        """Sample multiple opponents from the pool.

        Args:
            n: Number of opponents to sample.
            method: Sampling method.
            allow_duplicates: Whether the same checkpoint can be sampled multiple times.

        Returns:
            List of sampled CheckpointInfo objects.
        """
        if not self.checkpoints:
            return []

        if allow_duplicates:
            return [self.sample_opponent(method) for _ in range(n)]
        else:
            # Sample without replacement
            sampled = []
            exclude_ids = set()
            for _ in range(min(n, len(self.checkpoints))):
                opponent = self.sample_opponent(method, exclude_ids=exclude_ids)
                if opponent:
                    sampled.append(opponent)
                    exclude_ids.add(opponent.checkpoint_id)
            return sampled

    def update_checkpoint_stats(
        self,
        checkpoint_id: str,
        win_rate: Optional[float] = None,
        elo: Optional[float] = None,
        games_played_delta: int = 0,
    ) -> None:
        """Update statistics for a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to update.
            win_rate: New win rate against current policy.
            elo: New Elo rating.
            games_played_delta: Number of new games played.
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                if win_rate is not None:
                    checkpoint.win_rate_vs_current = win_rate
                if elo is not None:
                    checkpoint.elo = elo
                    # Also update in Elo tracker if available
                    if self._elo_tracker is not None:
                        self._elo_tracker.set_rating(checkpoint_id, elo)
                checkpoint.games_played += games_played_delta
                break

        self._save_pool_state()

    def sync_elo_from_tracker(self) -> None:
        """Synchronize Elo ratings from the tracker to checkpoints.

        Call this after running evaluation matches to update checkpoint
        Elo ratings from the tracker's computed values.
        """
        if self._elo_tracker is None:
            return

        for checkpoint in self.checkpoints:
            new_elo = self._elo_tracker.get_rating(checkpoint.checkpoint_id)
            if new_elo != checkpoint.elo:
                checkpoint.elo = new_elo

        self._save_pool_state()

    def get_checkpoints_by_elo_tier(
        self,
        tier: str,
        n: int = 3,
    ) -> list[CheckpointInfo]:
        """Get checkpoints from a specific Elo tier.

        Args:
            tier: One of "top", "middle", "bottom".
            n: Number of checkpoints to return.

        Returns:
            List of checkpoints from the specified tier.
        """
        if not self.checkpoints:
            return []

        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.elo, reverse=True)

        if tier == "top":
            return sorted_checkpoints[:n]
        elif tier == "bottom":
            return sorted_checkpoints[-n:]
        elif tier == "middle":
            mid = len(sorted_checkpoints) // 2
            start = max(0, mid - n // 2)
            return sorted_checkpoints[start:start + n]
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[CheckpointInfo]:
        """Get a checkpoint by its ID."""
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the checkpoint with the highest Elo rating."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.elo)

    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the most recently saved checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.step)

    def best_elo(self) -> float:
        """Get the highest Elo in the pool."""
        if not self.checkpoints:
            return self.config.initial_elo
        return max(c.elo for c in self.checkpoints)

    def elo_spread(self) -> float:
        """Get the Elo spread (max - min) in the pool."""
        if len(self.checkpoints) < 2:
            return 0.0
        elos = [c.elo for c in self.checkpoints]
        return max(elos) - min(elos)

    def refresh(self) -> None:
        """Refresh pool state from disk.
        
        This is used by subprocess environments to pick up new checkpoints
        saved by the main training process. Reloads the pool_state.json file.
        """
        self._load_pool_state()

    def __len__(self) -> int:
        """Number of checkpoints in the pool."""
        return len(self.checkpoints)

    def __iter__(self):
        """Iterate over checkpoints."""
        return iter(self.checkpoints)

    def prime_from_pool(self, source_dir: str, n_top: int = 3, n_random: int = 2) -> int:
        """Prime this pool by copying selected checkpoints from an existing pool.

        Selects the top-N checkpoints by Elo and a random sample from the
        remainder (no duplicates), copies their model files into this pool's
        save directory, and registers them with preserved Elo ratings.

        win_rate_vs_current and games_played are intentionally reset — they
        carry no meaning in the context of a new training run.

        Args:
            source_dir: Path to the source opponent pool directory containing
                pool_state.json and checkpoint files.
            n_top: Number of highest-Elo checkpoints to copy.
            n_random: Number of additional randomly sampled checkpoints to copy.

        Returns:
            Number of checkpoints successfully copied.
        """
        source_dir = Path(source_dir)
        state_path = source_dir / "pool_state.json"
        if not state_path.exists():
            return 0

        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            source_checkpoints = [
                CheckpointInfo.from_dict(c) for c in state.get("checkpoints", [])
            ]
        except Exception:
            return 0

        # Validate source checkpoint files exist
        valid_source = []
        for ckpt in source_checkpoints:
            path = Path(ckpt.path)
            if path.exists() or path.with_suffix("").exists():
                valid_source.append(ckpt)

        if not valid_source:
            return 0

        # Top-N by Elo
        sorted_by_elo = sorted(valid_source, key=lambda c: c.elo, reverse=True)
        top = sorted_by_elo[:n_top]
        top_ids = {c.checkpoint_id for c in top}

        # Random sample from remainder (no duplicates with top)
        remainder = [c for c in valid_source if c.checkpoint_id not in top_ids]
        random_picks = random.sample(remainder, min(n_random, len(remainder)))

        selected = top + random_picks

        # Copy each checkpoint into this pool
        copied = 0
        for ckpt in selected:
            src_path = Path(ckpt.path)
            if not src_path.exists():
                src_path = src_path.with_suffix("")
                if not src_path.exists():
                    continue

            dst_path = self.save_dir / src_path.name
            shutil.copy2(str(src_path), str(dst_path))

            new_info = CheckpointInfo(
                checkpoint_id=ckpt.checkpoint_id,
                path=str(dst_path),
                step=ckpt.step,
                elo=ckpt.elo,
                created_at=ckpt.created_at,
                win_rate_vs_current=0.5,
                games_played=0,
                metadata=ckpt.metadata,
            )
            self.checkpoints.append(new_info)

            if self._elo_tracker is not None:
                self._elo_tracker.register_checkpoint(new_info.checkpoint_id, new_info.elo)

            copied += 1

        if copied > 0:
            self._save_pool_state()

        return copied

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _pfsp_weight(self, win_rate: float) -> float:
        """Compute PFSP sampling weight based on win rate.

        Opponents with ~50% win rate are most valuable for learning.
        """
        # f(x) = x(1-x) peaks at 0.5
        return win_rate * (1 - win_rate) + 0.1  # Small epsilon for exploration

    def _prune_pool(self) -> None:
        """Remove checkpoints to maintain pool size."""
        while len(self.checkpoints) > self.config.pool_size:
            if self.config.prune_strategy == "oldest":
                # Remove oldest checkpoint
                to_remove = min(self.checkpoints, key=lambda c: c.step)
            elif self.config.prune_strategy == "lowest_elo":
                # Remove lowest Elo checkpoint
                to_remove = min(self.checkpoints, key=lambda c: c.elo)
            elif self.config.prune_strategy == "least_diverse":
                # Remove checkpoint with most similar Elo to neighbors
                to_remove = self._find_least_diverse()
            else:
                to_remove = self.checkpoints[0]

            self._remove_checkpoint(to_remove)

    def _find_least_diverse(self) -> CheckpointInfo:
        """Find the checkpoint that adds least diversity (closest Elo to neighbors)."""
        if len(self.checkpoints) <= 1:
            return self.checkpoints[0]

        # Sort by Elo
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.elo)

        min_gap = float("inf")
        least_diverse = sorted_checkpoints[0]

        for i, checkpoint in enumerate(sorted_checkpoints):
            if i == 0:
                gap = sorted_checkpoints[1].elo - checkpoint.elo
            elif i == len(sorted_checkpoints) - 1:
                gap = checkpoint.elo - sorted_checkpoints[-2].elo
            else:
                # Gap to both neighbors
                gap = min(
                    checkpoint.elo - sorted_checkpoints[i - 1].elo,
                    sorted_checkpoints[i + 1].elo - checkpoint.elo,
                )

            if gap < min_gap:
                min_gap = gap
                least_diverse = checkpoint

        return least_diverse

    def _remove_checkpoint(self, checkpoint: CheckpointInfo) -> None:
        """Remove a checkpoint from the pool and disk."""
        # Remove from list
        self.checkpoints = [c for c in self.checkpoints if c.checkpoint_id != checkpoint.checkpoint_id]

        # Remove from Elo tracker if available
        if self._elo_tracker is not None:
            self._elo_tracker.remove_checkpoint(checkpoint.checkpoint_id)

        # Remove files from disk
        try:
            path = Path(checkpoint.path)
            if path.exists():
                path.unlink()
            # Also try without .zip extension
            path_no_ext = path.with_suffix("")
            if path_no_ext.exists():
                path_no_ext.unlink()
        except Exception:
            pass  # Ignore file removal errors

    def _save_pool_state(self) -> None:
        """Save pool state to disk."""
        state_path = self.save_dir / "pool_state.json"
        state = {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "config": {
                "pool_size": self.config.pool_size,
                "save_interval": self.config.save_interval,
                "min_elo_gap": self.config.min_elo_gap,
                "initial_elo": self.config.initial_elo,
                "prune_strategy": self.config.prune_strategy,
            },
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_pool_state(self) -> None:
        """Load pool state from disk."""
        state_path = self.save_dir / "pool_state.json"
        if not state_path.exists():
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)

            self.checkpoints = [
                CheckpointInfo.from_dict(c) for c in state.get("checkpoints", [])
            ]

            # Validate checkpoint files still exist
            valid_checkpoints = []
            for checkpoint in self.checkpoints:
                path = Path(checkpoint.path)
                if path.exists() or path.with_suffix("").exists():
                    valid_checkpoints.append(checkpoint)
            self.checkpoints = valid_checkpoints

        except Exception:
            # If loading fails, start with empty pool
            self.checkpoints = []

    def refresh(self) -> None:
        """Refresh pool state from disk.

        Call this to pick up new checkpoints added by other processes.
        Useful for SubprocVecEnv workers to see checkpoints saved by
        the main training process.
        """
        self._load_pool_state()
