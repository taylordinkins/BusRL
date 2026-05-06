"""Training script for AlphaZero-style MCTS agent on the Bus board game.

Separate from the MaskablePPO pipeline (scripts/train.py). Checkpoints are
saved as plain .pt + _config.json files (incompatible with PPO .zip format).

Smoke test (should complete in <5 minutes):
    python scripts/train_mcts.py \\
        --iterations 2 --games_per_iter 2 --n_simulations 20 --n_workers 1
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rl.bus_env import BusEnv
from rl.config import ObservationConfig
from rl.hierarchical_action_space import HierarchicalActionMapping, HeadId
from rl.alphazero_network import AlphaZeroNetwork, AlphaZeroNetworkConfig
from rl.mcts import MCTSConfig
from rl.alphazero_trainer import AlphaZeroTrainer, AlphaZeroTrainingConfig
from data.loader import load_default_board


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_env_factory(num_players: int, obs_config: ObservationConfig):
    """Return a picklable factory that produces a reset-ready BusEnv."""

    def _factory() -> BusEnv:
        env = BusEnv(num_players=num_players, obs_config=obs_config)
        env.reset()
        return env

    return _factory


def build_head_catalog_sizes(board) -> dict[int, int]:
    """Map HeadId int → action catalog size from HierarchicalActionMapping."""
    mapping = HierarchicalActionMapping(board)
    return {head.value: mapping.head_size(head) for head in HeadId}


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ── Observation config ────────────────────────────────────────────────────
    obs_config = ObservationConfig(
        use_slot_actionability=args.use_slot_actionability,
    )
    print(f"Observation dim: {obs_config.total_observation_dim}")

    # ── Action space ──────────────────────────────────────────────────────────
    board = load_default_board()
    tmp_env = BusEnv(num_players=args.num_players, obs_config=obs_config)
    tmp_env.reset()
    num_actions = tmp_env.action_space.n          # = max_head_actions
    print(f"Action space size (max_head_actions): {num_actions}")

    head_catalog_sizes = build_head_catalog_sizes(board) if args.use_per_phase_heads else None

    # ── Network ───────────────────────────────────────────────────────────────
    net_config = AlphaZeroNetworkConfig(
        obs_dim=obs_config.total_observation_dim,
        num_actions=num_actions,
        num_players=args.num_players,
        trunk_layers=args.trunk_layers,
        use_per_phase_heads=args.use_per_phase_heads,
        trunk_activation="relu",
        use_layer_norm=True,
    )

    if args.initial_checkpoint:
        print(f"Loading initial checkpoint: {args.initial_checkpoint}")
        network = AlphaZeroNetwork.load(args.initial_checkpoint)
        network.to(device)
    else:
        network = AlphaZeroNetwork(net_config, head_catalog_sizes=head_catalog_sizes)
        network.to(device)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {param_count:,}")

    # ── MCTS config ───────────────────────────────────────────────────────────
    mcts_config = MCTSConfig(
        n_simulations=args.n_simulations,
        c_puct=args.c_puct,
        temperature=1.0,
        temperature_threshold=args.temperature_threshold,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        num_players=args.num_players,
    )

    # ── Training config ───────────────────────────────────────────────────────
    train_config = AlphaZeroTrainingConfig(
        games_per_iteration=args.games_per_iter,
        n_self_play_workers=args.n_workers,
        replay_buffer_size=args.replay_buffer_size,
        min_buffer_size=min(args.min_buffer_size, args.replay_buffer_size),
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_schedule="cosine",
        weight_decay=1e-4,
        max_grad_norm=args.max_grad_norm,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        eval_games=args.eval_games,
        eval_rank_threshold=args.eval_rank_threshold,
        eval_every_n_iterations=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_iterations=args.save_every,
        use_reward_shaping=args.use_reward_shaping,
        tensorboard=args.tensorboard,
        self_play_verbose=True,
        self_play_progress_every=args.self_play_progress_every,
    )

    env_factory = make_env_factory(args.num_players, obs_config)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = AlphaZeroTrainer(
        network=network,
        training_config=train_config,
        env_factory=env_factory,
        mcts_config=mcts_config,
        device=device,
    )

    if args.initial_checkpoint:
        trainer._prev_checkpoint_path = args.initial_checkpoint

    trainer.train(n_iterations=args.iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AlphaZero agent for Bus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Environment ───────────────────────────────────────────────────────────
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--use_slot_actionability", action="store_true",
                        help="Enable per-slot actionability obs feature")

    # ── Training loop ─────────────────────────────────────────────────────────
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of self-play + train cycles")
    parser.add_argument("--games_per_iter", type=int, default=100,
                        help="Self-play games per iteration")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel self-play workers (1 = sequential)")
    parser.add_argument("--train_steps", type=int, default=1000,
                        help="Gradient update steps per iteration")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--replay_buffer_size", type=int, default=200_000)
    parser.add_argument("--min_buffer_size", type=int, default=10_000,
                        help="Do not train until buffer reaches this size")

    # ── Network ───────────────────────────────────────────────────────────────
    parser.add_argument("--trunk_layers", type=int, nargs="+", default=[512, 512, 256],
                        help="Hidden layer sizes for shared trunk MLP")
    parser.add_argument("--use_per_phase_heads", action="store_true",
                        help="Use per-phase policy heads instead of flat head")

    # ── MCTS ─────────────────────────────────────────────────────────────────
    parser.add_argument("--n_simulations", type=int, default=400,
                        help="MCTS simulations per move")
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--temperature_threshold", type=int, default=30,
                        help="Switch to greedy selection after this many moves")

    # ── Optimizer ────────────────────────────────────────────────────────────
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ── Evaluation ───────────────────────────────────────────────────────────
    parser.add_argument("--eval_games", type=int, default=20)
    parser.add_argument("--eval_rank_threshold", type=float, default=0.55)
    parser.add_argument("--eval_every", type=int, default=5)

    # ── Checkpointing ────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_dir", type=str, default="logs/alphazero")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--initial_checkpoint", type=str, default=None,
                        help="Resume from an AlphaZeroNetwork .pt file")

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use_reward_shaping", action="store_true",
                        help="Blend step rewards into terminal value targets")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--self_play_progress_every", type=int, default=5,
                        help="When verbose, print every N completed self-play games")

    args = parser.parse_args()
    main(args)
