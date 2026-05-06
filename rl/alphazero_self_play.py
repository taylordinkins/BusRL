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


def _compute_rank_vector(env) -> np.ndarray:
    """Compute rank-based value targets from final game state.

    z[i] = (num_players - rank[i]) / (num_players - 1)
    Tied players share the average of their ranks.
    """
    state = env._engine.state
    scores = {p.player_id: p.score for p in state.players}
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


class SelfPlayWorker:
    """Plays a single complete game and produces labeled SelfPlaySamples."""

    def __init__(
        self,
        network,          # AlphaZeroNetwork
        env_factory: Callable,  # () -> BusEnv (must return a reset-ready env)
        mcts_config,      # MCTSConfig
        game_id: int = 0,
        use_reward_shaping: bool = False,
    ):
        self.network = network
        self.env_factory = env_factory
        self.mcts_config = mcts_config
        self.game_id = game_id
        self.use_reward_shaping = use_reward_shaping

    def play_game(self) -> list[SelfPlaySample]:
        """Play one game; return a list of labeled samples."""
        from .mcts import AlphaZeroMCTS

        mcts = AlphaZeroMCTS(self.network, self.mcts_config)
        env = self.env_factory()
        env.reset()

        # Accumulate (obs, head_id, visit_dist, player_id, move_number, step_reward)
        partial: list[tuple[np.ndarray, int, np.ndarray, int, int, float]] = []
        move_number = 0
        cumulative_rewards: dict[int, float] = {}  # for optional reward shaping

        while True:
            if env._engine is None or env._engine.is_game_over():
                break

            decision = env._get_decision_context()
            head_id_obj = decision.get("head_id") if decision else None
            head_id = head_id_obj.value if head_id_obj is not None else -1
            current_player = env.get_current_player()
            obs = env._get_observation().copy()

            action, visit_dist = mcts.search_with_policy(env, move_number=move_number)

            _, reward, terminated, truncated, _ = env.step(action)
            cumulative_rewards[current_player] = (
                cumulative_rewards.get(current_player, 0.0) + float(reward)
            )

            partial.append((obs, head_id, visit_dist, current_player, move_number, float(reward)))
            move_number += 1

            if terminated or truncated:
                break

        # Compute terminal value vector from final scores
        if env._engine is not None:
            z = _compute_rank_vector(env)
        else:
            n = self.mcts_config.num_players
            z = np.full(n, 1.0 / n, dtype=np.float32)

        # Optionally blend terminal outcome with accumulated step rewards
        if self.use_reward_shaping and cumulative_rewards:
            max_abs = max(abs(v) for v in cumulative_rewards.values()) or 1.0
            for pid, r in cumulative_rewards.items():
                z[pid] = 0.9 * z[pid] + 0.1 * (r / max_abs * 0.5 + 0.5)
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

        return samples


def run_self_play_parallel(
    network,
    env_factory: Callable,
    mcts_config,
    n_games: int,
    n_workers: int = 1,
    use_reward_shaping: bool = False,
    verbose: bool = False,
    progress_every: int = 1,
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

    if n_workers <= 1:
        total_moves = 0
        for game_id in range(n_games):
            worker = SelfPlayWorker(
                network, env_factory, mcts_config,
                game_id=game_id,
                use_reward_shaping=use_reward_shaping,
            )
            samples = worker.play_game()
            all_samples.extend(samples)
            total_moves += len(samples)
            completed = game_id + 1
            if verbose and (completed % progress_every == 0 or completed == n_games):
                avg_moves = total_moves / max(completed, 1)
                print(
                    f"  Self-play progress: {completed}/{n_games} games, "
                    f"last={len(samples)} moves, avg={avg_moves:.1f}"
                )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run(game_id: int) -> list[SelfPlaySample]:
            worker = SelfPlayWorker(
                network, env_factory, mcts_config,
                game_id=game_id,
                use_reward_shaping=use_reward_shaping,
            )
            return worker.play_game()

        completed = 0
        total_moves = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run, gid): gid for gid in range(n_games)}
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    completed += 1
                    total_moves += len(samples)
                    if verbose and (completed % progress_every == 0 or completed == n_games):
                        avg_moves = total_moves / max(completed, 1)
                        print(
                            f"  Self-play progress: {completed}/{n_games} games, "
                            f"last={len(samples)} moves, avg={avg_moves:.1f}"
                        )
                except Exception as exc:
                    print(f"  Game {gid} failed: {exc}")

    return all_samples
