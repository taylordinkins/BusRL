"""Collect behavioral cloning (BC) game records from a pretrained MaskablePPO model.

Runs N complete games using the PPO policy (greedy by default) and records
(obs, head_id, action_probs, mask, value_target) for every step. At game end,
computes rank-based value targets (z) and attaches them to all steps from that game.

The output pickle is directly compatible with scripts/bc_pretrain.py.

Usage:
    python scripts/collect_bc_data.py \\
        --ppo_checkpoint logs/ppo_bus_run1/best_pool_model/best_model.zip \\
        --n_games 1000 \\
        --output logs/bc_data/bc_data.pkl \\
        --use_slot_actionability \\
        --device auto
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rl.bus_env import BusEnv
from rl.config import ObservationConfig
from rl.alphazero_self_play import _compute_rank_vector, get_final_scores


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_ppo_model(checkpoint_path: str, env: BusEnv, device: torch.device):
    """Load MaskablePPO model with graceful error handling."""
    from sb3_contrib import MaskablePPO

    print(f"Loading PPO model from: {checkpoint_path}")
    try:
        model = MaskablePPO.load(checkpoint_path, env=env, device=device)
    except Exception as e:
        # Fallback: load without env (some SB3 versions require env for custom_objects)
        print(f"  Load with env failed ({e}), retrying without env...")
        model = MaskablePPO.load(checkpoint_path, device=device)
    model.policy.eval()
    print(f"  Loaded. Policy device: {next(model.policy.parameters()).device}")
    return model


def get_ppo_probs(
    ppo_model,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Extract masked action probability distribution from PPO policy.

    Returns:
        probs: np.ndarray, shape (max_head_actions,).
            Valid (unmasked) indices have nonzero probability.
            Probabilities sum to ~1 over valid indices.
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        dist = ppo_model.policy.get_distribution(obs_t, action_masks=mask)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()

    # Guard against NaN (can occur with all-masked logits)
    if np.isnan(probs).any() or probs.sum() <= 0:
        valid = np.where(mask)[0]
        probs = np.zeros_like(probs)
        if len(valid) > 0:
            probs[valid] = 1.0 / len(valid)

    return probs


def play_game(
    ppo_model,
    env: BusEnv,
    game_id: int,
    device: torch.device,
    greedy: bool = True,
) -> tuple[list[dict], dict]:
    """Play one complete game with the PPO policy.

    Returns:
        records:   List of per-step dicts (value_target filled in after game).
        game_info: Summary dict with 'scores', 'z', 'n_moves', 'terminated'.
    """
    env.reset()
    step_records: list[dict] = []
    move_number = 0
    terminated = False

    while True:
        if env._engine is None or env._engine.is_game_over():
            terminated = True
            break

        obs = env._get_observation().copy()
        mask = env.action_masks()
        decision = env._get_decision_context()
        head_id_obj = decision.get("head_id") if decision else None
        head_id = head_id_obj.value if head_id_obj is not None else -1
        current_player = env.get_current_player()

        probs = get_ppo_probs(ppo_model, obs, mask, device)

        if greedy:
            # Argmax over valid actions only
            valid = np.where(mask)[0]
            action = int(valid[int(np.argmax(probs[valid]))])
        else:
            # Stochastic: sample proportional to masked probs
            valid = np.where(mask)[0]
            valid_probs = probs[valid].astype(np.float64)
            total = valid_probs.sum()
            if total <= 0:
                action = int(np.random.choice(valid))
            else:
                valid_probs /= total
                action = int(np.random.choice(valid, p=valid_probs))

        step_records.append({
            "obs": obs,
            "head_id": head_id,
            "action_probs": probs.astype(np.float32),
            "mask": mask,
            "player_id": current_player,
            "game_id": game_id,
            "move_number": move_number,
            # value_target filled in after game ends
        })

        _, _, step_terminated, step_truncated, _ = env.step(action)
        move_number += 1

        if step_terminated:
            terminated = True
            break
        if step_truncated:
            break

    # Compute terminal rank vector z
    try:
        z = _compute_rank_vector(env)
    except Exception as e:
        n = env.num_players if hasattr(env, "num_players") else 4
        print(f"    Warning: _compute_rank_vector failed ({e}); using uniform z.")
        z = np.full(n, 1.0 / n, dtype=np.float32)

    # Attach z to every step record
    for record in step_records:
        record["value_target"] = z.copy()

    # Collect final scores for reporting
    scores: dict[int, float] = {}
    try:
        scores = get_final_scores(env)
    except Exception:
        pass

    game_info = {
        "terminated": terminated,
        "scores": scores,
        "z": z,
        "n_moves": move_number,
    }
    return step_records, game_info


def collect_bc_data(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    print(f"Device: {device}")

    obs_config = ObservationConfig(use_slot_actionability=args.use_slot_actionability)
    print(f"Observation dim: {obs_config.total_observation_dim}")
    print(f"use_slot_actionability: {args.use_slot_actionability}")

    env = BusEnv(num_players=args.num_players, obs_config=obs_config)
    env.reset()
    print(f"Action space size: {env.action_space.n}")

    ppo_model = load_ppo_model(args.ppo_checkpoint, env, device)

    dataset: list[dict] = []
    total_moves = 0
    n_terminated = 0
    all_scores: list[float] = []

    t_start = time.time()
    print(f"\nCollecting {args.n_games} games (greedy={args.greedy}) ...")

    for game_id in range(args.n_games):
        records, game_info = play_game(
            ppo_model, env, game_id, device, greedy=args.greedy
        )
        dataset.extend(records)
        total_moves += game_info["n_moves"]
        if game_info["terminated"]:
            n_terminated += 1
        all_scores.extend(game_info["scores"].values())

        # Progress reporting
        completed = game_id + 1
        every = max(1, args.n_games // 20)
        if completed % every == 0 or completed == args.n_games:
            elapsed = time.time() - t_start
            rate = completed / elapsed
            z_str = ", ".join(f"{v:.2f}" for v in game_info["z"])
            sc_str = ", ".join(str(s) for s in game_info["scores"].values())
            print(
                f"  [{completed}/{args.n_games}] "
                f"moves={game_info['n_moves']}, "
                f"scores=[{sc_str}], z=[{z_str}], "
                f"rate={rate:.1f} games/s"
            )

    elapsed = time.time() - t_start
    print(f"\nCollection complete in {elapsed:.1f}s")
    print(f"  Games: {args.n_games} ({n_terminated} terminated naturally)")
    print(f"  Total moves: {total_moves} ({total_moves / args.n_games:.1f} avg/game)")
    if all_scores:
        print(
            f"  Scores: min={min(all_scores):.1f}, max={max(all_scores):.1f}, "
            f"mean={np.mean(all_scores):.2f}"
        )
        unique = sorted(set(all_scores))
        print(f"  Unique score values: {len(unique)} (first few: {unique[:10]})")

    # Save dataset
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(dataset, f, protocol=4)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(dataset)} records to {output} ({size_mb:.1f} MB)")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect BC game records from a pretrained MaskablePPO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ppo_checkpoint", type=str, required=True,
        help="Path to MaskablePPO .zip checkpoint file",
    )
    parser.add_argument(
        "--n_games", type=int, default=1000,
        help="Number of games to collect",
    )
    parser.add_argument(
        "--output", type=str, default="logs/bc_data/bc_data.pkl",
        help="Output pickle file path",
    )
    parser.add_argument(
        "--num_players", type=int, default=4,
    )
    parser.add_argument(
        "--use_slot_actionability", action="store_true",
        help="Must match the flag used during PPO training",
    )
    parser.add_argument(
        "--greedy", action="store_true", default=True,
        help="Use argmax action selection (greedy). Use --no-greedy for stochastic.",
    )
    parser.add_argument(
        "--no-greedy", dest="greedy", action="store_false",
        help="Use stochastic action selection (sample from PPO distribution)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    args = parser.parse_args()
    collect_bc_data(args)


if __name__ == "__main__":
    main()
