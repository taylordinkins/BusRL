"""AlphaZero training loop for Bus game.

Alternates between:
  1. Self-play  — run games with MCTS, add samples to replay buffer
  2. Training   — gradient updates on buffered samples
  3. Evaluation — compare new vs old checkpoint; promote if better

Network, checkpoint, and replay-buffer paths are all separate from the
existing MaskablePPO pipeline (train.py / logs/ppo_*).
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class AlphaZeroTrainingConfig:
    # Self-play
    games_per_iteration: int = 100
    n_self_play_workers: int = 1

    # Replay buffer
    replay_buffer_size: int = 200_000
    min_buffer_size: int = 10_000

    # Training
    train_steps_per_iteration: int = 1_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    lr_schedule: str = "cosine"       # "cosine" | "constant"
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Evaluation
    eval_games: int = 20
    eval_rank_threshold: float = 0.55  # promote if avg normalized rank > threshold
    eval_every_n_iterations: int = 5

    # Checkpointing
    checkpoint_dir: str = "logs/alphazero"
    save_every_n_iterations: int = 5

    # Reward
    use_reward_shaping: bool = False   # False = terminal outcome only (AlphaZero default)

    # TensorBoard
    tensorboard: bool = False
    self_play_verbose: bool = False
    self_play_progress_every: int = 5


class AlphaZeroTrainer:
    """Orchestrates self-play, training, and evaluation for AlphaZero."""

    def __init__(
        self,
        network,              # AlphaZeroNetwork
        training_config: AlphaZeroTrainingConfig,
        env_factory: Callable,  # () -> BusEnv
        mcts_config,          # MCTSConfig
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.config = training_config
        self.env_factory = env_factory
        self.mcts_config = mcts_config
        self.device = device or next(network.parameters()).device

        self.network.to(self.device)

        self.optimizer = torch.optim.Adam(
            network.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        from .alphazero_self_play import ReplayBuffer
        self.replay_buffer = ReplayBuffer(max_size=training_config.replay_buffer_size)

        self._iteration = 0
        self._checkpoint_dir = Path(training_config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        if training_config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self._checkpoint_dir / "tb"))
            except ImportError:
                print("TensorBoard not available; continuing without it.")

        self._prev_checkpoint_path: Optional[str] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def train(self, n_iterations: int) -> None:
        """Run n_iterations of self-play → train → (eval) cycles."""
        print(f"AlphaZero training: {n_iterations} iterations")
        print(f"  Checkpoint dir: {self._checkpoint_dir}")
        print(f"  Device: {self.device}")
        if self._prev_checkpoint_path is None:
            self.save_checkpoint(0, name="incumbent_init", set_as_reference=True)
        else:
            print(f"  Incumbent checkpoint: {self._prev_checkpoint_path}")

        # LR scheduler covers the full training run
        if self.config.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=n_iterations * self.config.train_steps_per_iteration,
            )
        else:
            scheduler = None

        for it in range(1, n_iterations + 1):
            self._iteration = it
            t0 = time.time()

            # ── Self-play ────────────────────────────────────────────────────
            print(f"\n[Iter {it}] Self-play ({self.config.games_per_iteration} games)...")
            samples = self._self_play_phase()
            self.replay_buffer.add_game(samples)
            print(f"  Buffer size: {len(self.replay_buffer)}")

            if len(self.replay_buffer) < self.config.min_buffer_size:
                print(f"  Buffer below min ({self.config.min_buffer_size}), skipping training.")
                continue

            # ── Training ─────────────────────────────────────────────────────
            print(f"[Iter {it}] Training ({self.config.train_steps_per_iteration} steps)...")
            loss_info = self._training_phase(scheduler)
            print(
                f"  policy_loss={loss_info['policy_loss']:.4f}  "
                f"value_loss={loss_info['value_loss']:.4f}  "
                f"total={loss_info['total_loss']:.4f}"
            )

            if self._writer is not None:
                global_step = it * self.config.train_steps_per_iteration
                self._writer.add_scalar("train/policy_loss", loss_info["policy_loss"], global_step)
                self._writer.add_scalar("train/value_loss", loss_info["value_loss"], global_step)
                self._writer.add_scalar("train/total_loss", loss_info["total_loss"], global_step)

            # ── Evaluation ───────────────────────────────────────────────────
            if it % self.config.eval_every_n_iterations == 0 and self._prev_checkpoint_path:
                print(f"[Iter {it}] Evaluating against previous checkpoint...")
                avg_rank = self._evaluation_phase()
                promoted = avg_rank >= self.config.eval_rank_threshold
                print(
                    f"  avg_normalized_rank={avg_rank:.4f}  "
                    f"threshold={self.config.eval_rank_threshold}  "
                    f"{'✓ PROMOTED' if promoted else '✗ not promoted'}"
                )
                if promoted:
                    self.save_checkpoint(it, name="incumbent", set_as_reference=True)
                else:
                    print("  Reverting challenger to incumbent checkpoint.")
                    self.load_checkpoint(self._prev_checkpoint_path)

                if self._writer is not None:
                    self._writer.add_scalar("eval/avg_rank", avg_rank, it)

            # ── Checkpoint ───────────────────────────────────────────────────
            if it % self.config.save_every_n_iterations == 0:
                self.save_checkpoint(it, set_as_reference=False)

            elapsed = time.time() - t0
            print(f"[Iter {it}] Done in {elapsed:.1f}s")

        # Final save
        self.save_checkpoint(self._iteration, name="final", set_as_reference=False)
        if self._writer is not None:
            self._writer.close()

    # ── Phases ────────────────────────────────────────────────────────────────

    def _self_play_phase(self):
        """Run self-play games and return flat list of samples."""
        from .alphazero_self_play import run_self_play_parallel

        self.network.eval()
        samples = run_self_play_parallel(
            network=self.network,
            env_factory=self.env_factory,
            mcts_config=self.mcts_config,
            n_games=self.config.games_per_iteration,
            n_workers=self.config.n_self_play_workers,
            use_reward_shaping=self.config.use_reward_shaping,
            verbose=self.config.self_play_verbose,
            progress_every=self.config.self_play_progress_every,
        )
        return samples

    def _training_phase(self, scheduler=None) -> dict:
        """Gradient updates on replay buffer; returns average losses."""
        self.network.train()
        policy_losses, value_losses, total_losses = [], [], []

        for _ in range(self.config.train_steps_per_iteration):
            batch = self.replay_buffer.sample_batch(self.config.batch_size)
            if not batch:
                break

            p_loss, v_loss = self._compute_loss(batch)
            total = (
                self.config.policy_loss_weight * p_loss
                + self.config.value_loss_weight * v_loss
            )

            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            policy_losses.append(p_loss.item())
            value_losses.append(v_loss.item())
            total_losses.append(total.item())

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        }

    def _evaluation_phase(self) -> float:
        """Play eval_games against the previous checkpoint; return avg normalized rank.

        Uses the same tied-rank formula as training (average rank across tied players).
        The new model rotates through all player seats across games to remove positional
        bias. Returns a value in [0, 1]; random play yields ≈ 0.5 in a 4-player game.
        """
        from .mcts import AlphaZeroMCTS, MCTSConfig
        from .alphazero_network import AlphaZeroNetwork

        if self._prev_checkpoint_path is None:
            return 0.0

        old_net = AlphaZeroNetwork.load(self._prev_checkpoint_path)
        old_net.to(self.device)
        old_net.eval()

        # No Dirichlet noise and greedy action selection for deterministic eval
        eval_mcts_cfg = MCTSConfig(
            n_simulations=self.mcts_config.n_simulations,
            c_puct=self.mcts_config.c_puct,
            temperature=0.0,
            dirichlet_epsilon=0.0,
            num_players=self.mcts_config.num_players,
        )
        new_mcts = AlphaZeroMCTS(self.network, eval_mcts_cfg)
        old_mcts = AlphaZeroMCTS(old_net, eval_mcts_cfg)

        num_players = self.mcts_config.num_players
        rank_sum = 0.0

        self.network.eval()
        for game_idx in range(self.config.eval_games):
            # env_factory() already calls reset() internally
            env = self.env_factory()
            new_model_slot = game_idx % num_players

            while True:
                if env._engine is None or env._engine.is_game_over():
                    break
                cp = env.get_current_player()
                mcts = new_mcts if cp == new_model_slot else old_mcts
                action = mcts.search(env)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            if env._engine is not None:
                rank_sum += self._normalized_rank(
                    env, player_id=new_model_slot
                )

        return rank_sum / max(self.config.eval_games, 1)

    @staticmethod
    def _normalized_rank(env, player_id: int) -> float:
        """Compute z = (n - rank) / (n - 1) using average rank for ties.

        Matches the formula used in AlphaZeroMCTS._get_terminal_values().
        """
        state = env._engine.state
        scores = {p.player_id: p.score for p in state.players}
        n = len(scores)
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])

        i = 0
        while i < n:
            j = i
            while j < n and sorted_items[j][1] == sorted_items[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                if sorted_items[k][0] == player_id:
                    return (n - avg_rank) / max(n - 1, 1)
            i = j

        return 0.0

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy cross-entropy and value MSE for one batch.

        Policy loss: -mean(sum(π_mcts * log_softmax(logits)))
        Value loss:  mean(||v_θ - z||²)

        In per-phase mode we group samples by head_id and process each
        group separately (different head sizes require separate forward calls).
        """
        if self.network.config.use_per_phase_heads:
            return self._compute_loss_per_phase(batch)

        # ── Flat mode ─────────────────────────────────────────────────────────
        obs_t = torch.as_tensor(
            np.stack([s.obs for s in batch]), dtype=torch.float32, device=self.device
        )
        pi_t = torch.as_tensor(
            np.stack([s.policy_target for s in batch]), dtype=torch.float32, device=self.device
        )
        z_t = torch.as_tensor(
            np.stack([s.value_target for s in batch]), dtype=torch.float32, device=self.device
        )

        logits, value = self.network(obs_t)  # (B, num_actions), (B, num_players)

        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -torch.mean(torch.sum(pi_t * log_probs, dim=1))
        value_loss = F.mse_loss(value, z_t)

        return policy_loss, value_loss

    def _compute_loss_per_phase(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss for per-phase head mode: process each head_id group separately."""
        from collections import defaultdict

        groups: dict[int, list] = defaultdict(list)
        for s in batch:
            groups[s.head_id].append(s)

        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        n_total = 0

        for head_id, group in groups.items():
            if head_id < 0:
                continue  # unknown head_id, skip

            obs_t = torch.as_tensor(
                np.stack([s.obs for s in group]), dtype=torch.float32, device=self.device
            )
            z_t = torch.as_tensor(
                np.stack([s.value_target for s in group]), dtype=torch.float32, device=self.device
            )

            head_size = self.network.policy_heads[head_id].out_features
            pi_t = torch.as_tensor(
                np.stack([s.policy_target[:head_size] for s in group]),
                dtype=torch.float32, device=self.device,
            )

            logits, value = self.network(obs_t, head_id=head_id)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = policy_loss + torch.sum(-torch.sum(pi_t * log_probs, dim=1))
            value_loss = value_loss + torch.sum((value - z_t) ** 2)
            n_total += len(group)

        if n_total == 0:
            return policy_loss, value_loss

        policy_loss = policy_loss / n_total
        value_loss = value_loss / n_total
        return policy_loss, value_loss

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        iteration: int,
        name: Optional[str] = None,
        set_as_reference: bool = True,
    ) -> str:
        """Save network + optimizer state; return path to the .pt file."""
        stem = name if name else f"checkpoint_{iteration:04d}"
        pt_path = self._checkpoint_dir / f"{stem}.pt"
        meta_path = self._checkpoint_dir / f"{stem}_meta.json"

        self.network.save(pt_path)
        torch.save(self.optimizer.state_dict(), self._checkpoint_dir / f"{stem}_optim.pt")

        meta = {
            "iteration": iteration,
            "buffer_size": len(self.replay_buffer),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if set_as_reference:
            self._prev_checkpoint_path = str(pt_path)
        print(f"  Saved checkpoint: {pt_path}")
        return str(pt_path)

    def load_checkpoint(self, path: str) -> None:
        """Load network weights (and optionally optimizer) from a checkpoint."""
        from .alphazero_network import AlphaZeroNetwork

        net = AlphaZeroNetwork.load(path)
        self.network.load_state_dict(net.state_dict())
        self._prev_checkpoint_path = path

        optim_path = Path(path).parent / (Path(path).stem + "_optim.pt")
        if optim_path.exists():
            self.optimizer.load_state_dict(
                torch.load(str(optim_path), map_location=self.device, weights_only=True)
            )
        print(f"Loaded checkpoint: {path}")
