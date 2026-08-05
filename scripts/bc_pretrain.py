"""Behavioral cloning (BC) pre-training for AlphaZeroNetwork.

Trains a fresh AlphaZeroNetwork via supervised imitation from BC game records
collected by scripts/collect_bc_data.py. The resulting .pt checkpoint is
compatible with --initial_checkpoint in scripts/train_mcts.py.

Supports both flat and per-phase policy head modes:
  Flat (default):     Single Linear(trunk, num_actions) head; PPO flat
                      distribution used directly as target.
  Per-phase:          10 heads (one per HeadId); PPO flat probs trimmed to
  (--use_per_phase_heads)  each head's size and renormalized.

Policy loss:
    Soft cross-entropy: -sum(ppo_probs * log_softmax(logits)), averaged over
    all samples in the batch. In per-phase mode, averaged across head groups
    weighted by number of samples per head.

Value loss:
    MSE between value head output (shape: num_players) and rank vector z.

Usage (flat mode):
    python scripts/bc_pretrain.py \\
        --bc_data logs/bc_data/bc_data.pkl \\
        --output logs/bc_warmstart/network.pt \\
        --epochs 10 --batch_size 512 --device auto

Usage (per-phase mode):
    python scripts/bc_pretrain.py \\
        --bc_data logs/bc_data/bc_data.pkl \\
        --output logs/bc_warmstart/network.pt \\
        --epochs 10 --batch_size 512 \\
        --use_per_phase_heads --use_slot_actionability --device auto
"""

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from rl.bus_env import BusEnv
from rl.config import ObservationConfig
from rl.hierarchical_action_space import HierarchicalActionMapping, HeadId
from rl.alphazero_network import AlphaZeroNetwork, AlphaZeroNetworkConfig
from data.loader import load_default_board


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_bc_data(path: str) -> list[dict]:
    print(f"Loading BC data from: {path}")
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} records")
    sample = dataset[0]
    required = {"obs", "head_id", "action_probs", "mask", "value_target"}
    missing = required - set(sample.keys())
    if missing:
        raise ValueError(f"BC data missing keys: {missing}")
    print(
        f"  obs_dim={sample['obs'].shape[0]}, "
        f"num_actions={sample['action_probs'].shape[0]}, "
        f"num_players={sample['value_target'].shape[0]}"
    )
    return dataset


def build_head_catalog_sizes(board) -> dict[int, int]:
    mapping = HierarchicalActionMapping(board)
    return {head.value: mapping.head_size(head) for head in HeadId}


# ── Loss functions ────────────────────────────────────────────────────────────

def compute_loss_per_phase(
    network: AlphaZeroNetwork,
    batch: list[dict],
    device: torch.device,
    policy_loss_weight: float,
    value_loss_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Per-phase policy head mode.

    Single trunk pass for the full batch; policy forward done per head_id group.
    PPO flat probs are trimmed to each head's size and renormalized.

    Returns:
        (total_loss, policy_loss_scalar, value_loss_scalar)
    """
    obs_np = np.stack([s["obs"] for s in batch])
    z_np = np.stack([s["value_target"] for s in batch])
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device)

    # Single trunk pass — gradients flow through trunk for both heads
    features = network.trunk(obs_t)                   # (B, trunk_dim)
    values = network.value_head(features)             # (B, num_players)
    value_loss = F.mse_loss(values, z_t)

    # Per-head policy losses
    head_ids = [s["head_id"] for s in batch]
    unique_heads = sorted(set(h for h in head_ids if h >= 0))
    policy_loss_terms: list[tuple[int, torch.Tensor]] = []

    for h in unique_heads:
        idx = [i for i, hid in enumerate(head_ids) if hid == h]
        if not idx:
            continue

        head_size = network.policy_heads[h].out_features

        # Trim PPO flat probs to this head's size; renormalize
        target_np = np.stack([batch[i]["action_probs"][:head_size] for i in idx])
        row_sums = target_np.sum(axis=1, keepdims=True)
        target_np = target_np / np.maximum(row_sums, 1e-9)

        mask_np = np.stack([batch[i]["mask"][:head_size] for i in idx])
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)

        # Use pre-computed trunk features for this head group
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        features_h = features[idx_t]                              # (n_h, trunk_dim)
        logits_h = network.policy_heads[h](features_h)            # (n_h, head_size)
        logits_h = logits_h.masked_fill(~mask_t, -1e9)

        log_probs_h = F.log_softmax(logits_h, dim=-1)
        # Soft CE: -sum(target * log_prob) averaged over samples
        policy_loss_h = -(target_t * log_probs_h).sum(dim=-1).mean()
        policy_loss_terms.append((len(idx), policy_loss_h))

    if policy_loss_terms:
        n_total = sum(n for n, _ in policy_loss_terms)
        # Weighted average by number of samples per head
        policy_loss = sum(n * loss for n, loss in policy_loss_terms) / n_total
    else:
        policy_loss = torch.tensor(0.0, device=device)

    total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
    return total_loss, float(policy_loss.item()), float(value_loss.item())


def compute_loss_flat(
    network: AlphaZeroNetwork,
    batch: list[dict],
    device: torch.device,
    policy_loss_weight: float,
    value_loss_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Flat policy head mode.

    Full PPO distribution (max_head_actions,) used as target directly.

    Returns:
        (total_loss, policy_loss_scalar, value_loss_scalar)
    """
    obs_np = np.stack([s["obs"] for s in batch])
    z_np = np.stack([s["value_target"] for s in batch])
    target_np = np.stack([s["action_probs"] for s in batch])
    mask_np = np.stack([s["mask"] for s in batch])

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    features = network.trunk(obs_t)
    values = network.value_head(features)
    value_loss = F.mse_loss(values, z_t)

    logits = network.policy_head(features)
    logits = logits.masked_fill(~mask_t, -1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(target_t * log_probs).sum(dim=-1).mean()

    total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
    return total_loss, float(policy_loss.item()), float(value_loss.item())


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    network: AlphaZeroNetwork,
    val_data: list[dict],
    device: torch.device,
    batch_size: int,
    use_per_phase_heads: bool,
    policy_loss_weight: float,
    value_loss_weight: float,
) -> tuple[float, float, float]:
    """Compute validation losses without gradient updates."""
    network.eval()
    loss_fn = compute_loss_per_phase if use_per_phase_heads else compute_loss_flat
    total_pol, total_val, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for start in range(0, len(val_data), batch_size):
            batch = val_data[start : start + batch_size]
            if not batch:
                continue
            _, pol, val = loss_fn(
                network, batch, device, policy_loss_weight, value_loss_weight
            )
            total_pol += pol
            total_val += val
            n_batches += 1

    network.train()
    if n_batches == 0:
        return 0.0, 0.0, 0.0
    avg_pol = total_pol / n_batches
    avg_val = total_val / n_batches
    avg_total = policy_loss_weight * avg_pol + value_loss_weight * avg_val
    return avg_total, avg_pol, avg_val


# ── Main training function ────────────────────────────────────────────────────

def train_bc(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Mode: {'per-phase heads' if args.use_per_phase_heads else 'flat head'}")

    # ── Load BC data ──────────────────────────────────────────────────────────
    dataset = load_bc_data(args.bc_data)
    random.shuffle(dataset)

    val_n = max(1, int(len(dataset) * args.val_split))
    val_data = dataset[-val_n:]
    train_data = dataset[:-val_n]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Infer dims from BC data ───────────────────────────────────────────────
    bc_obs_dim = dataset[0]["obs"].shape[0]
    num_actions = dataset[0]["action_probs"].shape[0]
    num_players = dataset[0]["value_target"].shape[0]

    obs_config = ObservationConfig(use_slot_actionability=args.use_slot_actionability)
    expected_obs_dim = obs_config.total_observation_dim
    if bc_obs_dim != expected_obs_dim:
        raise ValueError(
            f"BC data obs_dim={bc_obs_dim} does not match "
            f"ObservationConfig obs_dim={expected_obs_dim}. "
            f"Ensure --use_slot_actionability matches the collection flags."
        )
    print(f"obs_dim={bc_obs_dim}, num_actions={num_actions}, num_players={num_players}")

    # ── Build network ─────────────────────────────────────────────────────────
    head_catalog_sizes: dict[int, int] | None = None
    if args.use_per_phase_heads:
        board = load_default_board()
        head_catalog_sizes = build_head_catalog_sizes(board)
        print(f"Head catalog sizes: {head_catalog_sizes}")

    net_config = AlphaZeroNetworkConfig(
        obs_dim=bc_obs_dim,
        num_actions=num_actions,
        num_players=num_players,
        trunk_layers=args.trunk_layers,
        use_per_phase_heads=args.use_per_phase_heads,
        trunk_activation="relu",
        use_layer_norm=True,
    )
    network = AlphaZeroNetwork(net_config, head_catalog_sizes=head_catalog_sizes)
    network.to(device)
    network.train()

    n_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = compute_loss_per_phase if args.use_per_phase_heads else compute_loss_flat

    # ── Output directory ──────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.epochs} epochs, batch_size={args.batch_size} ...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_data)
        epoch_pol, epoch_val, n_batches = 0.0, 0.0, 0
        t_epoch = time.time()

        for start in range(0, len(train_data), args.batch_size):
            batch = train_data[start : start + args.batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            total_loss, pol_loss, val_loss = loss_fn(
                network, batch, device,
                args.policy_loss_weight, args.value_loss_weight,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_pol += pol_loss
            epoch_val += val_loss
            n_batches += 1

        avg_pol = epoch_pol / max(n_batches, 1)
        avg_val = epoch_val / max(n_batches, 1)
        avg_total = args.policy_loss_weight * avg_pol + args.value_loss_weight * avg_val
        elapsed = time.time() - t_epoch

        val_total, val_pol, val_val = evaluate(
            network, val_data, device, args.batch_size,
            args.use_per_phase_heads, args.policy_loss_weight, args.value_loss_weight,
        )

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train: total={avg_total:.4f} pol={avg_pol:.4f} val={avg_val:.4f} | "
            f"val:   total={val_total:.4f} pol={val_pol:.4f} val={val_val:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint by validation loss
        if val_total < best_val_loss:
            best_val_loss = val_total
            network.save(output)
            print(f"  -> Saved best checkpoint (val_loss={val_total:.4f}) to {output}")

    # Save final state regardless
    final_path = output.parent / (output.stem + "_final.pt")
    network.save(final_path)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Best checkpoint:  {output} (val_loss={best_val_loss:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BC pre-training of AlphaZeroNetwork from PPO game records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--bc_data", type=str, required=True,
        help="BC data pickle file from collect_bc_data.py",
    )
    parser.add_argument(
        "--output", type=str, default="logs/bc_warmstart/network.pt",
        help="Output .pt path (best checkpoint by validation loss)",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.05,
        help="Fraction of data held out for validation loss monitoring",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs over the full dataset")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--policy_loss_weight", type=float, default=1.0)
    parser.add_argument("--value_loss_weight", type=float, default=1.0)

    # ── Network ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--trunk_layers", type=int, nargs="+", default=[512, 512, 256],
        help="Must match trunk_layers used in train_mcts.py",
    )
    parser.add_argument(
        "--use_per_phase_heads", action="store_true",
        help="Use 10 per-phase policy heads instead of a single flat head",
    )
    parser.add_argument(
        "--use_slot_actionability", action="store_true",
        help="Must match the flag used during BC collection",
    )

    # ── Device ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
    )

    args = parser.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
