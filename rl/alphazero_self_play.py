"""Self-play data collection for AlphaZero training.

Runs complete Bus games with MCTS and produces labeled (obs, policy, value)
samples. The value target z is the terminal rank vector computed from final
scores, applied uniformly to every move of the game (same terminal outcome
for all players is known at game end).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SelfPlaySample:
    """A single labeled training sample from a self-play game."""

    obs: np.ndarray            # shape (obs_dim,)
    head_id: int               # active HeadId int value (-1 if unknown)
    policy_target: np.ndarray  # MCTS visit distribution, shape (num_actions,)
    value_target: np.ndarray   # rank vector z, shape (num_players,), each in [0, 1]
    player_id: int             # player who acted at this step
    game_id: int
    move_number: int


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for SelfPlaySamples."""

    def __init__(self, max_size: int = 200_000):
        self._buffer: deque[SelfPlaySample] = deque(maxlen=max_size)

    def add_game(self, samples: list[SelfPlaySample]) -> None:
        for s in samples:
            self._buffer.append(s)

    def sample_batch(self, batch_size: int) -> list[SelfPlaySample]:
        n = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), n)

    def __len__(self) -> int:
        return len(self._buffer)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(list(self._buffer), f, protocol=4)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data: list[SelfPlaySample] = pickle.load(f)
        self._buffer = deque(data, maxlen=self._buffer.maxlen)


def get_final_scores(env) -> dict[int, float]:
    """Return the canonical final scores used for AlphaZero targets/eval."""
    state = env._engine.state
    return {p.player_id: p.get_final_score() for p in state.players}


def _compute_rank_vector(env) -> np.ndarray:
    """Compute rank-based value targets from final game state.

    Uses each player's final score (deliveries minus time stone penalties)
    so that time stone differences break ties when no deliveries occur,
    giving the value head a weak but real signal during bootstrap.

    z[i] = (num_players - rank[i]) / (num_players - 1)
    Tied players share the average of their ranks.
    """
    scores = get_final_scores(env)
    n = len(scores)
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])

    ranks: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[sorted_items[k][0]] = avg_rank
        i = j

    denom = max(n - 1, 1)
    value_target = np.zeros(n, dtype=np.float32)
    for pid, rank in ranks.items():
        value_target[pid] = (n - rank) / denom

    return value_target


def normalized_rank_for_player(env, player_id: int) -> float:
    """Return the normalized tied rank for a specific player."""
    scores = get_final_scores(env)
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


class SelfPlayWorker:
    """Plays a single complete game and produces labeled SelfPlaySamples."""

    def __init__(
        self,
        network,          # AlphaZeroNetwork
        env_factory: Callable,  # () -> BusEnv (must return a reset-ready env)
        mcts_config,      # MCTSConfig
        game_id: int = 0,
        use_reward_shaping: bool = False,
        bc_network=None,  # optional frozen BC network for mid-game curriculum (5.1)
        bc_rounds: int = 0,  # play this many rounds with BC policy before handing to MCTS
    ):
        self.network = network
        self.env_factory = env_factory
        self.mcts_config = mcts_config
        self.game_id = game_id
        self.use_reward_shaping = use_reward_shaping
        self.bc_network = bc_network
        self.bc_rounds = bc_rounds

    def play_game(self) -> tuple[list[SelfPlaySample], dict]:
        """Play one game; return (samples, game_info).

        game_info keys:
            terminated (bool): True if game ended naturally, False if truncated.
            scores (dict):     {player_id: score} at game end.
            z (np.ndarray):    Rank-based value targets, shape (num_players,).
            n_moves (int):     Number of moves played.
        """
        from .mcts import AlphaZeroMCTS

        mcts = AlphaZeroMCTS(self.network, self.mcts_config)
        env = self.env_factory()
        env.reset()

        # Accumulate (obs, head_id, visit_dist, player_id, move_number, step_reward)
        partial: list[tuple[np.ndarray, int, np.ndarray, int, int, float]] = []
        move_number = 0
        cumulative_rewards: dict[int, float] = {}  # for optional reward shaping
        terminated = False

        # Mid-game curriculum (5.1): play the first bc_rounds rounds with the
        # BC policy (greedy, no MCTS, no samples stored) so that MCTS starts
        # from states where deliveries are imminent.
        if self.bc_network is not None and self.bc_rounds > 0:
            while True:
                if env._engine is None or env._engine.is_game_over():
                    terminated = True
                    break
                round_num = env._engine.state.global_state.round_number
                if round_num > self.bc_rounds:
                    break
                obs = env._get_observation()
                mask = env.action_masks()
                decision = env._get_decision_context()
                head_id_obj = decision.get("head_id") if decision else None
                head_id = head_id_obj.value if head_id_obj is not None else None
                priors = self.bc_network.get_policy_priors(obs, mask, head_id=head_id)
                valid = np.where(mask)[0]
                if len(valid) == 0:
                    break
                action = int(valid[int(np.argmax(priors[valid]))])
                _, _, step_term, step_trunc, _ = env.step(action)
                if step_term or step_trunc:
                    terminated = True
                    break

        if terminated:
            # Game ended during BC warmup phase — return empty samples
            game_info = {"terminated": True, "scores": {}, "z": np.zeros(self.mcts_config.num_players, dtype=np.float32), "n_moves": 0}
            return [], game_info

        while True:
            if env._engine is None or env._engine.is_game_over():
                terminated = True
                break

            decision = env._get_decision_context()
            head_id_obj = decision.get("head_id") if decision else None
            head_id = head_id_obj.value if head_id_obj is not None else -1
            current_player = env.get_current_player()
            obs = env._get_observation().copy()

            action, visit_dist = mcts.search_with_policy(env, move_number=move_number)

            _, reward, step_term, step_trunc, _ = env.step(action)
            cumulative_rewards[current_player] = (
                cumulative_rewards.get(current_player, 0.0) + float(reward)
            )

            partial.append((obs, head_id, visit_dist, current_player, move_number, float(reward)))
            move_number += 1

            if step_term:
                terminated = True
                break
            if step_trunc:
                break

        # Compute terminal value vector from final scores
        scores: dict[int, float] = {}
        if env._engine is not None:
            z = _compute_rank_vector(env)
            try:
                scores = get_final_scores(env)
            except Exception:
                pass
        else:
            n = self.mcts_config.num_players
            z = np.full(n, 1.0 / n, dtype=np.float32)

        # Optionally blend terminal outcome with accumulated step rewards.
        # We rank players by their cumulative shaped rewards (same formula as z)
        # rather than normalizing by within-game max, which washes out the signal
        # when all players accumulate near-equal rewards (common during bootstrap).
        if self.use_reward_shaping and cumulative_rewards:
            n = len(cumulative_rewards)
            sorted_items = sorted(cumulative_rewards.items(), key=lambda x: -x[1])
            shaped_ranks: dict[int, float] = {}
            i = 0
            while i < n:
                j = i
                while j < n and sorted_items[j][1] == sorted_items[i][1]:
                    j += 1
                avg_rank = (i + 1 + j) / 2
                for k in range(i, j):
                    shaped_ranks[sorted_items[k][0]] = avg_rank
                i = j
            denom = max(n - 1, 1)
            alpha = 0.15  # weight on shaped signal; tune down in later runs
            for pid, rank in shaped_ranks.items():
                shaped_z = (n - rank) / denom
                z[pid] = (1 - alpha) * z[pid] + alpha * shaped_z
            z = np.clip(z, 0.0, 1.0)

        samples: list[SelfPlaySample] = []
        for obs, head_id, visit_dist, pid, move_num, _ in partial:
            samples.append(SelfPlaySample(
                obs=obs,
                head_id=head_id,
                policy_target=visit_dist,
                value_target=z.copy(),
                player_id=pid,
                game_id=self.game_id,
                move_number=move_num,
            ))

        game_info = {
            "terminated": terminated,
            "scores": scores,
            "z": z,
            "n_moves": move_number,
        }
        return samples, game_info


def run_self_play_parallel(
    network,
    env_factory: Callable,
    mcts_config,
    n_games: int,
    n_workers: int = 1,
    use_reward_shaping: bool = False,
    verbose: bool = False,
    progress_every: int = 1,
    bc_network=None,
    bc_rounds: int = 0,
) -> list[SelfPlaySample]:
    """Run n_games self-play games, sequentially or in parallel.

    Args:
        network:            AlphaZeroNetwork (CPU or GPU).
        env_factory:        Callable returning a fresh, reset-ready BusEnv.
        mcts_config:        MCTSConfig controlling search parameters.
        n_games:            Total number of games to play.
        n_workers:          Number of parallel workers.
                            >1 uses ThreadPoolExecutor (shares network weights).
        use_reward_shaping: Blend step rewards into z targets.
        verbose:            Print progress.
        progress_every:     Print every N completed games when verbose=True.

    Returns:
        Flat list of SelfPlaySamples from all games.
    """
    all_samples: list[SelfPlaySample] = []
    progress_every = max(int(progress_every), 1)

    n_terminated = 0
    n_truncated = 0
    all_scores: list[dict] = []   # one dict per game: {player_id: score}

    if n_workers <= 1:
        total_moves = 0
        for game_id in range(n_games):
            worker = SelfPlayWorker(
                network, env_factory, mcts_config,
                game_id=game_id,
                use_reward_shaping=use_reward_shaping,
                bc_network=bc_network,
                bc_rounds=bc_rounds,
            )
            samples, game_info = worker.play_game()
            all_samples.extend(samples)
            total_moves += len(samples)
            if game_info["terminated"]:
                n_terminated += 1
            else:
                n_truncated += 1
            if game_info["scores"]:
                all_scores.append(game_info["scores"])
            completed = game_id + 1
            if verbose and (completed % progress_every == 0 or completed == n_games):
                avg_moves = total_moves / max(completed, 1)
                score_str = ""
                if game_info["scores"]:
                    sv = list(game_info["scores"].values())
                    score_str = (
                        f", scores=[{', '.join(str(s) for s in sv)}]"
                        f" z=[{', '.join(f'{v:.2f}' for v in game_info['z'])}]"
                    )
                print(
                    f"  Self-play progress: {completed}/{n_games} games, "
                    f"last={len(samples)} moves, avg={avg_moves:.1f}{score_str}"
                )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run(game_id: int) -> tuple[list[SelfPlaySample], dict]:
            worker = SelfPlayWorker(
                network, env_factory, mcts_config,
                game_id=game_id,
                use_reward_shaping=use_reward_shaping,
                bc_network=bc_network,
                bc_rounds=bc_rounds,
            )
            return worker.play_game()

        completed = 0
        total_moves = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run, gid): gid for gid in range(n_games)}
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    samples, game_info = future.result()
                    all_samples.extend(samples)
                    completed += 1
                    total_moves += len(samples)
                    if game_info["terminated"]:
                        n_terminated += 1
                    else:
                        n_truncated += 1
                    if game_info["scores"]:
                        all_scores.append(game_info["scores"])
                    if verbose and (completed % progress_every == 0 or completed == n_games):
                        avg_moves = total_moves / max(completed, 1)
                        print(
                            f"  Self-play progress: {completed}/{n_games} games, "
                            f"last={len(samples)} moves, avg={avg_moves:.1f}"
                        )
                except Exception as exc:
                    print(f"  Game {gid} failed: {exc}")

    if verbose:
        print(
            f"  Termination: {n_terminated} natural, {n_truncated} truncated"
        )
        if all_scores:
            all_score_vals = [s for d in all_scores for s in d.values()]
            unique_scores = sorted(set(all_score_vals))
            print(
                f"  Score distribution: min={min(all_score_vals):.1f}"
                f"  max={max(all_score_vals):.1f}"
                f"  unique={len(unique_scores)}"
                f"  (first few: {unique_scores[:8]})"
            )
            # Delivery rate: use the best player score per game as a proxy (2.3)
            max_scores = [max(d.values()) for d in all_scores]
            print(
                f"  Deliveries per game: mean={sum(max_scores)/len(max_scores):.2f}"
                f"  min={min(max_scores)}  max={max(max_scores)}"
                f"  zero_delivery_games={sum(1 for s in max_scores if s == 0)}/{len(max_scores)}"
            )

    return all_samples
