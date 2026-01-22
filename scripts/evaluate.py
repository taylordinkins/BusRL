"""Evaluation script for Bus RL agent.

Supports both direct policy evaluation and MCTS-enhanced evaluation.

Usage:
    # Basic evaluation (policy only)
    python scripts/evaluate.py path/to/model.zip --num-games 10

    # MCTS-enhanced evaluation
    python scripts/evaluate.py path/to/model.zip --num-games 10 \
        --use-mcts --mcts-simulations 100

    # Compare MCTS vs policy-only
    python scripts/evaluate.py path/to/model.zip --compare-mcts
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# Add project root to sys.path to allow importing from rl module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from rl.bus_env import BusEnv
from rl.wrappers import BusEnvSelfPlayWrapper


def evaluate(args):
    """Run evaluation games."""
    print(f"Evaluating model: {args.model_path}")
    print(f"Games: {args.num_games}, Players: {args.num_players}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskablePPO.load(args.model_path, device=device)

    # Create player based on mode
    if args.use_mcts:
        from rl.mcts_player import MCTSPlayer
        from rl.mcts import MCTSConfig

        config = MCTSConfig(
            n_simulations=args.mcts_simulations,
            c_puct=args.mcts_c_puct,
            temperature=args.mcts_temperature,
            use_value_network=not args.mcts_use_rollout,
            dirichlet_alpha=args.mcts_dirichlet_alpha,
            dirichlet_epsilon=args.mcts_dirichlet_epsilon,
        )

        player = MCTSPlayer(model, config=config)
        print(f"\nUsing MCTS with {args.mcts_simulations} simulations")
        print(f"  c_puct: {args.mcts_c_puct}, temperature: {args.mcts_temperature}")
    else:
        player = None
        print("\nUsing direct policy inference")

    # Init env
    env = BusEnv(num_players=args.num_players)
    env = BusEnvSelfPlayWrapper(env)

    win_counts = {i: 0 for i in range(args.num_players)}
    total_scores = {i: 0 for i in range(args.num_players)}
    p_rewards = {i: 0.0 for i in range(args.num_players)}
    game_times = []

    for game_idx in range(args.num_games):
        start_time = time.time()
        obs, info = env.reset(seed=args.seed + game_idx if args.seed else None)
        terminated = False
        steps = 0

        while not terminated and steps < 5000:
            if args.use_mcts:
                # Get underlying BusEnv for MCTS
                base_env = env.env  # Unwrap BusEnvSelfPlayWrapper
                if hasattr(base_env, 'env'):  # Unwrap Monitor if present
                    base_env = base_env.env
                action = player.get_action(base_env)
            else:
                # Direct policy inference
                action_masks = env.action_masks()
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        game_time = time.time() - start_time
        game_times.append(game_time)

        # Record stats
        scores = info.get("scores", {})
        if scores:
            winner = max(scores, key=scores.get)
            win_counts[winner] += 1
            for p_id, score in scores.items():
                total_scores[p_id] += score

        p_rewards_game = info.get("episode_player_rewards", {})
        for p_id, rew in p_rewards_game.items():
            p_rewards[p_id] += rew

        print(f"Game {game_idx + 1}/{args.num_games} finished after {steps} steps "
              f"({game_time:.1f}s). Scores: {scores}")

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Mode: {'MCTS' if args.use_mcts else 'Policy Only'}")
    print(f"Total games: {args.num_games}")
    print(f"Average game time: {np.mean(game_times):.2f}s")
    print()

    for p_id in range(args.num_players):
        win_rate = win_counts[p_id] / args.num_games * 100
        avg_score = total_scores[p_id] / args.num_games
        avg_reward = p_rewards[p_id] / args.num_games
        print(f"Player {p_id}: Win Rate={win_rate:.1f}%, Avg Score={avg_score:.1f}, Avg Reward={avg_reward:.2f}")

    if args.use_mcts:
        stats = player.get_stats_summary()
        print(f"\nMCTS Statistics:")
        print(f"  Total searches: {stats['total_searches']}")
        print(f"  Avg simulations/search: {stats['avg_simulations_per_search']:.1f}")
        print(f"  Avg root value: {stats['avg_root_value']:.3f}")

    print("=" * 50)


def compare_mcts(args):
    """Compare MCTS vs policy-only performance."""
    print(f"Comparing MCTS vs Policy-only for: {args.model_path}")
    print(f"Games: {args.num_games}, Players: {args.num_players}")

    from rl.mcts_player import MCTSPlayer, PolicyPlayer, compare_players
    from rl.mcts import MCTSConfig

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskablePPO.load(args.model_path, device=device)

    # Create players
    config = MCTSConfig(
        n_simulations=args.mcts_simulations,
        c_puct=args.mcts_c_puct,
        temperature=args.mcts_temperature,
        use_value_network=not args.mcts_use_rollout,
    )

    mcts_player = MCTSPlayer(model, config=config)
    policy_player = PolicyPlayer(model, deterministic=True)

    # Create environment
    env = BusEnv(num_players=2)  # 2-player for head-to-head

    print(f"\nMCTS config: {args.mcts_simulations} sims, c_puct={args.mcts_c_puct}")
    print(f"Running {args.num_games} games...\n")

    results = compare_players(
        env=env,
        mcts_player=mcts_player,
        policy_player=policy_player,
        n_games=args.num_games,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"MCTS win rate: {results['mcts_win_rate']:.1%}")
    print(f"Policy win rate: {results['policy_win_rate']:.1%}")
    print(f"Draws: {results['draws']}")
    print(f"MCTS avg score: {results['mcts_avg_score']:.1f}")
    print(f"Policy avg score: {results['policy_avg_score']:.1f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Bus RL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("model_path", type=str, help="Path to the trained model .zip")

    # Basic options
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--num-players", type=int, default=4, help="Number of players in game")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    # MCTS options
    parser.add_argument("--use-mcts", action="store_true",
                        help="Use MCTS for action selection")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--mcts-c-puct", type=float, default=1.5,
                        help="MCTS exploration constant (PUCT)")
    parser.add_argument("--mcts-temperature", type=float, default=0.1,
                        help="Temperature for MCTS action selection (0=greedy)")
    parser.add_argument("--mcts-use-rollout", action="store_true",
                        help="Use random rollout instead of value network for leaf eval")
    parser.add_argument("--mcts-dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root exploration")
    parser.add_argument("--mcts-dirichlet-epsilon", type=float, default=0.0,
                        help="Weight of Dirichlet noise (0=disabled)")

    # Comparison mode
    parser.add_argument("--compare-mcts", action="store_true",
                        help="Run comparison between MCTS and policy-only")

    args = parser.parse_args()

    if args.compare_mcts:
        compare_mcts(args)
    else:
        evaluate(args)
