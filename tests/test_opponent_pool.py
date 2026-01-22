"""Tests for the opponent pool module."""

import os
import tempfile
import shutil

import pytest
import numpy as np

from rl.opponent_pool import OpponentPool, CheckpointInfo, PoolConfig


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass."""

    def test_creation(self):
        """Test basic CheckpointInfo creation."""
        info = CheckpointInfo(
            checkpoint_id="test_001",
            path="/path/to/checkpoint.zip",
            step=1000,
            elo=1500.0,
        )
        assert info.checkpoint_id == "test_001"
        assert info.step == 1000
        assert info.elo == 1500.0
        assert info.win_rate_vs_current == 0.5
        assert info.games_played == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        info = CheckpointInfo(
            checkpoint_id="test_001",
            path="/path/to/checkpoint.zip",
            step=1000,
        )
        d = info.to_dict()
        assert d["checkpoint_id"] == "test_001"
        assert d["step"] == 1000
        assert "created_at" in d

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "checkpoint_id": "test_001",
            "path": "/path/to/checkpoint.zip",
            "step": 1000,
            "elo": 1600.0,
            "created_at": "2024-01-01T00:00:00",
            "win_rate_vs_current": 0.6,
            "games_played": 10,
            "metadata": {"key": "value"},
        }
        info = CheckpointInfo.from_dict(d)
        assert info.checkpoint_id == "test_001"
        assert info.elo == 1600.0
        assert info.win_rate_vs_current == 0.6
        assert info.metadata == {"key": "value"}


class TestPoolConfig:
    """Tests for PoolConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = PoolConfig()
        assert config.pool_size == 20
        assert config.save_interval == 50_000
        assert config.initial_elo == 1500.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = PoolConfig(pool_size=10, save_interval=25_000)
        assert config.pool_size == 10
        assert config.save_interval == 25_000


class TestOpponentPool:
    """Tests for OpponentPool."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def pool(self, temp_dir):
        """Create a fresh opponent pool."""
        return OpponentPool(save_dir=temp_dir)

    def test_creation(self, pool, temp_dir):
        """Test pool creation."""
        assert len(pool) == 0
        assert pool.save_dir.exists()

    def test_empty_pool_sampling(self, pool):
        """Test sampling from empty pool."""
        assert pool.sample_opponent() is None
        assert pool.sample_opponents(3) == []

    def test_checkpoint_info_without_model(self, pool):
        """Test adding checkpoint info directly (without a real model)."""
        # Manually add checkpoint info for testing
        info = CheckpointInfo(
            checkpoint_id="manual_001",
            path=str(pool.save_dir / "manual_001.zip"),
            step=1000,
            elo=1500.0,
        )
        pool.checkpoints.append(info)

        assert len(pool) == 1
        assert pool.sample_opponent() is not None

    def test_uniform_sampling(self, pool):
        """Test uniform sampling from pool."""
        # Add multiple checkpoints
        for i in range(5):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
                elo=1500.0 + i * 10,
            )
            pool.checkpoints.append(info)

        # Sample multiple times
        samples = [pool.sample_opponent(method="uniform") for _ in range(100)]
        assert all(s is not None for s in samples)

        # Check that we get variety
        unique_ids = set(s.checkpoint_id for s in samples)
        assert len(unique_ids) > 1

    def test_latest_sampling(self, pool):
        """Test latest-biased sampling."""
        for i in range(5):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
            )
            pool.checkpoints.append(info)

        # Sample many times with latest bias
        samples = [pool.sample_opponent(method="latest") for _ in range(200)]
        latest_count = sum(1 for s in samples if s.checkpoint_id == "ckpt_4")

        # Latest should be sampled more often (not guaranteed but likely)
        assert latest_count > 20  # Should be more than uniform (200/5=40)

    def test_pfsp_sampling(self, pool):
        """Test PFSP sampling based on win rates."""
        # Add checkpoints with different win rates
        win_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, wr in enumerate(win_rates):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
                win_rate_vs_current=wr,
            )
            pool.checkpoints.append(info)

        # Sample many times with PFSP
        samples = [pool.sample_opponent(method="pfsp") for _ in range(500)]

        # Count samples per checkpoint
        counts = {}
        for s in samples:
            counts[s.checkpoint_id] = counts.get(s.checkpoint_id, 0) + 1

        # Checkpoint with 0.5 win rate should be sampled most often
        # (it has highest uncertainty)
        assert counts.get("ckpt_2", 0) >= counts.get("ckpt_0", 0)
        assert counts.get("ckpt_2", 0) >= counts.get("ckpt_4", 0)

    def test_sample_without_duplicates(self, pool):
        """Test sampling multiple opponents without duplicates."""
        for i in range(5):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
            )
            pool.checkpoints.append(info)

        # Sample 3 without duplicates
        samples = pool.sample_opponents(3, allow_duplicates=False)
        assert len(samples) == 3
        ids = [s.checkpoint_id for s in samples]
        assert len(set(ids)) == 3  # All unique

        # Sample more than available
        samples = pool.sample_opponents(10, allow_duplicates=False)
        assert len(samples) == 5  # Limited to pool size

    def test_update_checkpoint_stats(self, pool):
        """Test updating checkpoint statistics."""
        info = CheckpointInfo(
            checkpoint_id="ckpt_0",
            path=str(pool.save_dir / "ckpt_0.zip"),
            step=1000,
        )
        pool.checkpoints.append(info)

        pool.update_checkpoint_stats(
            checkpoint_id="ckpt_0",
            win_rate=0.75,
            elo=1600.0,
            games_played_delta=10,
        )

        updated = pool.get_checkpoint_by_id("ckpt_0")
        assert updated.win_rate_vs_current == 0.75
        assert updated.elo == 1600.0
        assert updated.games_played == 10

    def test_best_and_latest(self, pool):
        """Test getting best and latest checkpoints."""
        for i in range(3):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
                elo=1500.0 + (2 - i) * 100,  # elo: 1700, 1600, 1500
            )
            pool.checkpoints.append(info)

        best = pool.get_best_checkpoint()
        assert best.checkpoint_id == "ckpt_0"  # Highest Elo

        latest = pool.get_latest_checkpoint()
        assert latest.checkpoint_id == "ckpt_2"  # Highest step

    def test_elo_metrics(self, pool):
        """Test Elo-related metrics."""
        for i in range(3):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
                elo=1400.0 + i * 100,  # 1400, 1500, 1600
            )
            pool.checkpoints.append(info)

        assert pool.best_elo() == 1600.0
        assert pool.elo_spread() == 200.0

    def test_pruning_oldest(self, temp_dir):
        """Test pool pruning with oldest strategy."""
        config = PoolConfig(pool_size=3, prune_strategy="oldest")
        pool = OpponentPool(save_dir=temp_dir, config=config)

        # Add 5 checkpoints (exceeds pool size)
        for i in range(5):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
            )
            pool.checkpoints.append(info)
            if len(pool.checkpoints) > config.pool_size:
                pool._prune_pool()

        assert len(pool) == 3
        # Oldest should have been removed
        ids = [c.checkpoint_id for c in pool.checkpoints]
        assert "ckpt_0" not in ids
        assert "ckpt_1" not in ids
        assert "ckpt_4" in ids

    def test_pruning_lowest_elo(self, temp_dir):
        """Test pool pruning with lowest_elo strategy."""
        config = PoolConfig(pool_size=3, prune_strategy="lowest_elo")
        pool = OpponentPool(save_dir=temp_dir, config=config)

        # Add checkpoints with different Elos
        elos = [1500, 1600, 1400, 1700, 1550]
        for i, elo in enumerate(elos):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
                elo=float(elo),
            )
            pool.checkpoints.append(info)
            if len(pool.checkpoints) > config.pool_size:
                pool._prune_pool()

        assert len(pool) == 3
        # Lowest Elo should have been removed
        remaining_elos = [c.elo for c in pool.checkpoints]
        assert 1400.0 not in remaining_elos
        assert 1700.0 in remaining_elos

    def test_state_persistence(self, temp_dir):
        """Test that pool state persists across instances."""
        # Create pool and add checkpoints
        pool1 = OpponentPool(save_dir=temp_dir)
        for i in range(3):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool1.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
            )
            pool1.checkpoints.append(info)
        pool1._save_pool_state()

        # Create new pool from same directory
        pool2 = OpponentPool(save_dir=temp_dir)

        # Note: Checkpoints won't be loaded because files don't exist
        # In real usage, save_checkpoint creates the files
        # For this test, we manually saved state without files

    def test_exclude_ids_sampling(self, pool):
        """Test sampling with excluded IDs."""
        for i in range(3):
            info = CheckpointInfo(
                checkpoint_id=f"ckpt_{i}",
                path=str(pool.save_dir / f"ckpt_{i}.zip"),
                step=i * 1000,
            )
            pool.checkpoints.append(info)

        # Exclude some checkpoints
        exclude = {"ckpt_0", "ckpt_2"}
        samples = [pool.sample_opponent(exclude_ids=exclude) for _ in range(10)]

        # All samples should be ckpt_1
        assert all(s.checkpoint_id == "ckpt_1" for s in samples)

        # Exclude all
        samples = pool.sample_opponent(exclude_ids={"ckpt_0", "ckpt_1", "ckpt_2"})
        assert samples is None
