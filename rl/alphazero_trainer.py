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
import datetime
import json
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

    # KL regularization from BC checkpoint (3.2)
    kl_weight_initial: float = 0.0    # starting KL weight; 0.0 = disabled
    kl_total_iters: int = 100         # iterations over which kl_weight decays linearly to 0

    # Mid-game curriculum (5.1)
    bc_rounds: int = 0                # play this many rounds with BC policy before MCTS takes over

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
        bc_checkpoint_path: Optional[str] = None,
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
        self._optimizer_step_count = 0
        self._checkpoint_dir = Path(training_config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # BC network for KL regularization (3.2) and mid-game curriculum (5.1)
        self._bc_network = None
        if bc_checkpoint_path is not None:
            from .alphazero_network import AlphaZeroNetwork
            self._bc_network = AlphaZeroNetwork.load(bc_checkpoint_path)
            self._bc_network.to(self.device)
            self._bc_network.eval()
            print(f"Loaded BC network for KL reg / curriculum: {bc_checkpoint_path}")

        # Each run gets its own timestamped TensorBoard subdirectory so
        # restarts don't produce overlapping sawtooth curves (2.1)
        self._writer = None
        if training_config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self._writer = SummaryWriter(
                    log_dir=str(self._checkpoint_dir / "tb" / run_tag)
                )
            except ImportError:
                print("TensorBoard not available; continuing without it.")

        self._prev_checkpoint_path: Optional[str] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def train(self, n_iterations: int) -> None:
        """Run n_iterations of self-play → train → (eval) cycles."""
        start_iteration = self._iteration + 1
        end_iteration = self._iteration + n_iterations
        print(f"AlphaZero training: iterations {start_iteration}..{end_iteration}")
        print(f"  Checkpoint dir: {self._checkpoint_dir}")
        print(f"  Device: {self.device}")
        if self._prev_checkpoint_path is None:
            self.save_checkpoint(0, name="incumbent_init", set_as_reference=True)
        else:
            print(f"  Incumbent checkpoint: {self._prev_checkpoint_path}")

        # LR scheduler covers the full training run
        if self.config.lr_schedule == "cosine":
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", self.config.learning_rate)
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(
                    self._optimizer_step_count
                    + n_iterations * self.config.train_steps_per_iteration,
                    1,
                ),
                last_epoch=self._optimizer_step_count - 1,
            )
        else:
            scheduler = None

        for it in range(start_iteration, end_iteration + 1):
            self._iteration = it
            t0 = time.time()

            # ── Self-play ────────────────────────────────────────────────────
            print(f"\n[Iter {it}] Self-play ({self.config.games_per_iteration} games)...")
            samples = self._self_play_phase()
            self.replay_buffer.add_game(samples)
            if samples:
                z_arr = np.stack([s.value_target for s in samples])
                print(
                    f"  Value targets (z): mean={z_arr.mean():.4f}  std={z_arr.std():.4f}"
                    f"  min={z_arr.min():.4f}  max={z_arr.max():.4f}"
                    f"  unique_vals={len(np.unique(z_arr.round(3)))}"
                )
            print(f"  Buffer size: {len(self.replay_buffer)}")

            if len(self.replay_buffer) < self.config.min_buffer_size:
                print(f"  Buffer below min ({self.config.min_buffer_size}), skipping training.")
                continue

            # ── Training ─────────────────────────────────────────────────────
            print(f"[Iter {it}] Training ({self.config.train_steps_per_iteration} steps)...")
            loss_info = self._training_phase(scheduler)
            print(
                f"  policy_loss={loss_info['policy_loss']:.4f}  "
                f"value_loss={loss_info['value_loss']:.6f}  "
                f"total={loss_info['total_loss']:.4f}  "
                f"grad_norm={loss_info['grad_norm']:.4f}  "
                f"lr={loss_info['lr']:.2e}"
            )
            print(
                f"  z (target):  mean={loss_info['z_mean']:.4f}  std={loss_info['z_std']:.4f}"
                f"  |  v (pred):  mean={loss_info['v_mean']:.4f}  std={loss_info['v_std']:.4f}"
            )

            if self._writer is not None:
                global_step = it * self.config.train_steps_per_iteration
                self._writer.add_scalar("train/policy_loss", loss_info["policy_loss"], global_step)
                self._writer.add_scalar("train/value_loss", loss_info["value_loss"], global_step)
                self._writer.add_scalar("train/total_loss", loss_info["total_loss"], global_step)
                self._writer.add_scalar("train/grad_norm", loss_info["grad_norm"], global_step)
                self._writer.add_scalar("train/lr", loss_info["lr"], global_step)
                if not (loss_info["z_mean"] != loss_info["z_mean"]):  # not NaN
                    self._writer.add_scalar("train/z_mean", loss_info["z_mean"], global_step)
                    self._writer.add_scalar("train/z_std", loss_info["z_std"], global_step)
                    self._writer.add_scalar("train/v_mean", loss_info["v_mean"], global_step)
                    self._writer.add_scalar("train/v_std", loss_info["v_std"], global_step)

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
            bc_network=self._bc_network,
            bc_rounds=self.config.bc_rounds,
        )
        return samples

    def _training_phase(self, scheduler=None) -> dict:
        """Gradient updates on replay buffer; returns average losses."""
        self.network.train()
        policy_losses, value_losses, total_losses = [], [], []
        grad_norms = []
        z_means, z_stds, v_means, v_stds = [], [], [], []

        # Linear decay of KL weight from initial → 0 over kl_total_iters
        if self.config.kl_weight_initial > 0 and self.config.kl_total_iters > 0:
            decay = max(0.0, 1.0 - self._iteration / self.config.kl_total_iters)
            kl_weight = self.config.kl_weight_initial * decay
        else:
            kl_weight = 0.0

        for _ in range(self.config.train_steps_per_iteration):
            batch = self.replay_buffer.sample_batch(self.config.batch_size)
            if not batch:
                break

            p_loss, v_loss, diag = self._compute_loss(batch, kl_weight=kl_weight)
            total = (
                self.config.policy_loss_weight * p_loss
                + self.config.value_loss_weight * v_loss
            )

            self.optimizer.zero_grad()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self._optimizer_step_count += 1
            if scheduler is not None:
                scheduler.step()

            policy_losses.append(p_loss.item())
            value_losses.append(v_loss.item())
            total_losses.append(total.item())
            grad_norms.append(float(grad_norm))
            if diag:
                z_means.append(diag["z_mean"])
                z_stds.append(diag["z_std"])
                v_means.append(diag["v_mean"])
                v_stds.append(diag["v_std"])

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "lr": self.optimizer.param_groups[0]["lr"],
            "z_mean": float(np.mean(z_means)) if z_means else float("nan"),
            "z_std": float(np.mean(z_stds)) if z_stds else float("nan"),
            "v_mean": float(np.mean(v_means)) if v_means else float("nan"),
            "v_std": float(np.mean(v_stds)) if v_stds else float("nan"),
        }

    def _evaluation_phase(self) -> float:
        """Play eval_games against the previous checkpoint; return avg normalized rank.

        Uses the same tied-rank formula as training (average rank across tied players).
        The new model rotates through all player seats across games to remove positional
        bias. Returns a value in [0, 1]; random play yields ≈ 0.5 in a 4-player game.
        """
        from .mcts import AlphaZeroMCTS, MCTSConfig
        from .alphazero_network import AlphaZeroNetwork
        from .alphazero_self_play import normalized_rank_for_player

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
            temperature_threshold=self.mcts_config.temperature_threshold,
            dirichlet_alpha=self.mcts_config.dirichlet_alpha,
            dirichlet_epsilon=0.0,
            num_players=self.mcts_config.num_players,
            rollout_steps=self.mcts_config.rollout_steps,
            rollout_reward_weight=self.mcts_config.rollout_reward_weight,
            rollout_reward_scale=self.mcts_config.rollout_reward_scale,
            rollout_to_round_end=self.mcts_config.rollout_to_round_end,
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
                rank_sum += normalized_rank_for_player(env, player_id=new_model_slot)

        return rank_sum / max(self.config.eval_games, 1)

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(
        self, batch, kl_weight: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute policy cross-entropy and value MSE for one batch.

        Policy loss: -mean(sum(π_mcts * log_softmax(logits))) + kl_weight * KL(BC||current)
        Value loss:  mean(||v_θ - z||²)

        Returns (policy_loss, value_loss, diag) where diag contains z/v stats
        for debugging (empty dict in per-phase mode).

        In per-phase mode we group samples by head_id and process each
        group separately (different head sizes require separate forward calls).
        """
        if self.network.config.use_per_phase_heads:
            return self._compute_loss_per_phase(batch, kl_weight=kl_weight)

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

        if self._bc_network is not None and kl_weight > 0:
            with torch.no_grad():
                bc_logits, _ = self._bc_network(obs_t)
                bc_log_probs = F.log_softmax(bc_logits, dim=1)
            kl_loss = F.kl_div(log_probs, bc_log_probs.exp(), reduction="batchmean")
            policy_loss = policy_loss + kl_weight * kl_loss

        value_loss = F.mse_loss(value, z_t)

        with torch.no_grad():
            diag = {
                "z_mean": z_t.mean().item(),
                "z_std": z_t.std().item(),
                "v_mean": value.mean().item(),
                "v_std": value.std().item(),
            }

        return policy_loss, value_loss, diag

    def _compute_loss_per_phase(
        self, batch, kl_weight: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Loss for per-phase head mode: process each head_id group separately."""
        from collections import defaultdict

        groups: dict[int, list] = defaultdict(list)
        for s in batch:
            groups[s.head_id].append(s)

        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        n_total = 0
        all_z_list: list[torch.Tensor] = []
        all_v_list: list[torch.Tensor] = []

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

            if self._bc_network is not None and kl_weight > 0:
                with torch.no_grad():
                    bc_logits, _ = self._bc_network(obs_t, head_id=head_id)
                    bc_log_probs = F.log_softmax(bc_logits, dim=1)
                # KL(BC || current): penalizes moving away from BC prior
                # Use reduction="sum" so we can normalize consistently with policy_loss
                kl_loss = F.kl_div(log_probs, bc_log_probs.exp(), reduction="sum")
                policy_loss = policy_loss + kl_weight * kl_loss

            value_loss = value_loss + torch.sum((value - z_t) ** 2)
            n_total += len(group)
            all_z_list.append(z_t.detach())
            all_v_list.append(value.detach())

        if n_total == 0:
            return policy_loss, value_loss, {}

        policy_loss = policy_loss / n_total
        value_loss = value_loss / n_total

        with torch.no_grad():
            all_z = torch.cat(all_z_list, dim=0)
            all_v = torch.cat(all_v_list, dim=0)
            diag = {
                "z_mean": all_z.mean().item(),
                "z_std": all_z.std().item() if all_z.numel() > 1 else 0.0,
                "v_mean": all_v.mean().item(),
                "v_std": all_v.std().item() if all_v.numel() > 1 else 0.0,
            }
        return policy_loss, value_loss, diag

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
        replay_path = self._checkpoint_dir / f"{stem}_replay.pkl"

        self.network.save(pt_path)
        torch.save(self.optimizer.state_dict(), self._checkpoint_dir / f"{stem}_optim.pt")
        self.replay_buffer.save(str(replay_path))

        meta = {
            "iteration": iteration,
            "buffer_size": len(self.replay_buffer),
            "optimizer_step_count": self._optimizer_step_count,
            "replay_buffer_path": str(replay_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if set_as_reference:
            self._prev_checkpoint_path = str(pt_path)
        print(f"  Saved checkpoint: {pt_path}")
        return str(pt_path)

    def load_checkpoint(self, path: str) -> None:
        """Load network, optimizer, replay buffer, and iteration state if present."""
        from .alphazero_network import AlphaZeroNetwork

        net = AlphaZeroNetwork.load(path)
        self.network.load_state_dict(net.state_dict())
        self._prev_checkpoint_path = path

        optim_path = Path(path).parent / (Path(path).stem + "_optim.pt")
        if optim_path.exists():
            self.optimizer.load_state_dict(
                torch.load(str(optim_path), map_location=self.device, weights_only=True)
            )
        meta_path = Path(path).parent / (Path(path).stem + "_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._iteration = int(meta.get("iteration", self._iteration))
            self._optimizer_step_count = int(
                meta.get("optimizer_step_count", self._optimizer_step_count)
            )
            replay_path = meta.get("replay_buffer_path")
            if replay_path and Path(replay_path).exists():
                self.replay_buffer.load(replay_path)
        print(f"Loaded checkpoint: {path}")
