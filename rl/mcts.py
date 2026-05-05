"""AlphaZero-style MCTS for Bus game.

Full rewrite of the inference-only PPO placeholder. Key differences:

- Per-player value vector (shape: num_players) instead of a scalar
- Standalone AlphaZeroNetwork — no SB3 dependency
- Lazy env materialization: child envs are only cloned when first visited
- PUCT with sibling-level min-max Q normalization
- Temperature-based action selection with a configurable move threshold
- Correct multi-player backpropagation (all player slots updated each sim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .alphazero_network import AlphaZeroNetwork
    from .bus_env import BusEnv


@dataclass
class MCTSConfig:
    n_simulations: int = 400
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_threshold: int = 30   # greedy after this many moves
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25   # 0.0 disables noise (use for eval)
    num_players: int = 4


class MCTSNode:
    """Node in the AlphaZero MCTS tree.

    Children are created as virtual nodes (env=None) at expansion time and
    materialized (env cloned from parent) only when first selected as a leaf.
    This avoids cloning N envs per expansion.
    """

    __slots__ = (
        "env", "parent", "action_idx", "prior", "num_players",
        "visit_count", "value_sum", "children",
        "is_expanded", "is_terminal", "current_player",
    )

    def __init__(
        self,
        env: Optional["BusEnv"],
        parent: Optional["MCTSNode"],
        action_idx: int,
        prior: float,
        num_players: int,
    ):
        self.env = env                    # None until first visit
        self.parent = parent
        self.action_idx = action_idx
        self.prior = prior
        self.num_players = num_players

        self.visit_count: int = 0
        self.value_sum: np.ndarray = np.zeros(num_players, dtype=np.float64)

        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.current_player: int = 0      # player who acts FROM this state

    def q_value(self, player_id: int) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player_id]) / self.visit_count

    def ucb_score(
        self,
        c_puct: float,
        current_player: int,
        q_min: float,
        q_max: float,
    ) -> float:
        """PUCT score with min-max normalized Q."""
        q = self.q_value(current_player)
        q_norm = (q - q_min) / (q_max - q_min + 1e-8)
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )
        return q_norm + exploration

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Pick the child with the highest PUCT score.

        Q values are normalized over siblings (for the player acting at self)
        before computing UCB to keep the exploration term on a comparable scale.
        """
        cp = self.current_player
        q_vals = [child.q_value(cp) for child in self.children.values()]
        q_min = min(q_vals) if q_vals else 0.0
        q_max = max(q_vals) if q_vals else 1.0

        best_score = float("-inf")
        best_child: Optional["MCTSNode"] = None
        for child in self.children.values():
            score = child.ucb_score(c_puct, cp, q_min, q_max)
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_child

    def expand(self, priors: np.ndarray) -> None:
        """Create virtual children for every valid (unmasked) action.

        Only creates children for actions where the mask is True.
        Does NOT clone the env for children (lazy materialization).
        """
        assert self.env is not None, "Cannot expand a virtual node"
        mask = self.env.action_masks()
        for action_idx in np.where(mask)[0]:
            p = float(priors[action_idx]) if action_idx < len(priors) else 0.0
            self.children[int(action_idx)] = MCTSNode(
                env=None,
                parent=self,
                action_idx=int(action_idx),
                prior=p,
                num_players=self.num_players,
            )
        self.is_expanded = True


class AlphaZeroMCTS:
    """AlphaZero MCTS using an AlphaZeroNetwork for priors and value estimates.

    Per-player value backpropagation: every simulation propagates a
    (num_players,) value vector up the entire path. PUCT at each node uses
    only the value slot for the player who acts at that node.
    """

    def __init__(
        self,
        network: "AlphaZeroNetwork",
        config: Optional[MCTSConfig] = None,
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.device = next(network.parameters()).device

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, env: "BusEnv") -> int:
        """Run MCTS and return the greedy best action (no temperature)."""
        if self.config.n_simulations <= 0:
            action, _ = self._policy_only_action(env, move_number=self.config.temperature_threshold + 1)
            return action

        root = self._build_root(env, add_noise=(self.config.dirichlet_epsilon > 0))
        for _ in range(self.config.n_simulations):
            self._simulate(root)
        # Use a move number beyond the threshold to force greedy selection
        return self._select_action(root, move_number=self.config.temperature_threshold + 1)

    def search_with_policy(
        self,
        env: "BusEnv",
        move_number: int = 0,
    ) -> tuple[int, np.ndarray]:
        """Run MCTS and return (action, visit_distribution) for self-play training.

        Args:
            env:         Current environment state.
            move_number: Number of moves played so far (controls temperature).

        Returns:
            action:     Selected action index.
            visit_dist: Visit count distribution, shape (env.action_space.n,).
        """
        if self.config.n_simulations <= 0:
            return self._policy_only_action(env, move_number=move_number)

        root = self._build_root(env, add_noise=(self.config.dirichlet_epsilon > 0))
        for _ in range(self.config.n_simulations):
            self._simulate(root)

        action = self._select_action(root, move_number)

        num_actions = env.action_space.n
        visit_dist = np.zeros(num_actions, dtype=np.float32)
        total = sum(c.visit_count for c in root.children.values())
        if total > 0:
            for idx, child in root.children.items():
                if idx < num_actions:
                    visit_dist[idx] = child.visit_count / total

        return action, visit_dist

    # ── Internal methods ──────────────────────────────────────────────────────

    def _build_root(self, env: "BusEnv", add_noise: bool = True) -> MCTSNode:
        """Create, pre-expand, and optionally add Dirichlet noise to root."""
        root = MCTSNode(
            env=env.clone(),
            parent=None,
            action_idx=-1,
            prior=0.0,
            num_players=self.config.num_players,
        )
        root.current_player = root.env.get_current_player()

        if root.env._engine is None or root.env._engine.is_game_over():
            root.is_terminal = True
            return root

        priors, _ = self._get_priors_and_value(root.env)

        if add_noise:
            mask = root.env.action_masks()
            priors = self._add_dirichlet_noise(priors, mask)

        root.expand(priors)
        return root

    def _simulate(self, root: MCTSNode) -> None:
        """One full MCTS simulation: select → materialize → evaluate → backprop."""
        path: list[MCTSNode] = []
        node = root

        # Selection: walk to an unexpanded leaf
        while node.is_expanded and not node.is_terminal:
            path.append(node)
            node = node.select_child(self.config.c_puct)
        path.append(node)

        # Materialization: instantiate the node's env on first visit
        if node.env is None:
            parent = node.parent
            assert parent is not None and parent.env is not None
            node.env = parent.env.clone()
            _, _, terminated, truncated, _ = node.env.step(node.action_idx)
            node.is_terminal = terminated or truncated
            node.current_player = node.env.get_current_player()

        # Evaluation
        if node.is_terminal:
            values = self._get_terminal_values(node.env)
        else:
            priors, values = self._get_priors_and_value(node.env)
            if not node.is_expanded:
                node.expand(priors)

        # Backpropagation: all nodes on the path get the full value vector
        for n in path:
            n.visit_count += 1
            n.value_sum += values

    def _get_priors_and_value(
        self, env: "BusEnv"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the network for action priors and all-player value estimates."""
        obs = env._get_observation()
        mask = env.action_masks()
        decision = env._get_decision_context()
        head_id_obj = decision.get("head_id") if decision else None
        head_id = head_id_obj.value if head_id_obj is not None else None

        priors = self.network.get_policy_priors(obs, mask, head_id=head_id, device=self.device)
        values = self.network.get_value(obs, device=self.device)
        return priors, values

    def _get_terminal_values(self, env: "BusEnv") -> np.ndarray:
        """Rank-based value vector from final game scores.

        z[i] = (num_players - rank[i]) / (num_players - 1)
        Ties get the average rank of the tied positions.
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
            # 1-indexed average rank for positions i..j-1 (0-indexed in sorted order)
            avg_rank = (i + 1 + j) / 2
            for k in range(i, j):
                ranks[sorted_items[k][0]] = avg_rank
            i = j

        values = np.zeros(n, dtype=np.float32)
        denom = max(n - 1, 1)
        for player_id, rank in ranks.items():
            values[player_id] = (n - rank) / denom

        return values

    def _policy_only_action(
        self,
        env: "BusEnv",
        move_number: int,
    ) -> tuple[int, np.ndarray]:
        """Policy-only fallback for n_simulations <= 0."""
        mask = env.action_masks()
        valid = np.where(mask)[0]
        num_actions = env.action_space.n
        visit_dist = np.zeros(num_actions, dtype=np.float32)
        if len(valid) == 0:
            return 0, visit_dist

        priors, _ = self._get_priors_and_value(env)
        policy = np.zeros(num_actions, dtype=np.float64)
        width = min(len(priors), num_actions)
        policy[:width] = priors[:width]
        policy[~mask] = 0.0

        total = policy[valid].sum()
        if total <= 0:
            policy[valid] = 1.0 / len(valid)
        else:
            policy[valid] /= total

        if move_number > self.config.temperature_threshold or self.config.temperature <= 0:
            action = int(valid[int(np.argmax(policy[valid]))])
        elif self.config.temperature == 1.0:
            action = int(np.random.choice(valid, p=policy[valid]))
        else:
            tempered = np.zeros_like(policy)
            tempered[valid] = policy[valid] ** (1.0 / self.config.temperature)
            t_sum = tempered[valid].sum()
            if t_sum <= 0:
                action = int(np.random.choice(valid))
            else:
                tempered[valid] /= t_sum
                action = int(np.random.choice(valid, p=tempered[valid]))

        visit_dist[:] = policy.astype(np.float32)
        return action, visit_dist

    def _add_dirichlet_noise(
        self, priors: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Mix Dirichlet noise into the root priors for exploration."""
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return priors

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid))
        noisy = priors.copy()
        eps = self.config.dirichlet_epsilon
        for i, idx in enumerate(valid):
            noisy[idx] = (1 - eps) * priors[idx] + eps * noise[i]

        total = noisy[valid].sum()
        if total > 0:
            noisy[valid] /= total

        return noisy

    def _select_action(self, root: MCTSNode, move_number: int) -> int:
        """Choose action from root children via visit counts (with temperature)."""
        if not root.children:
            mask = root.env.action_masks()
            valid = np.where(mask)[0]
            return int(np.random.choice(valid)) if len(valid) > 0 else 0

        actions = list(root.children.keys())
        visits = np.array(
            [root.children[a].visit_count for a in actions], dtype=np.float64
        )

        if move_number > self.config.temperature_threshold or self.config.temperature == 0:
            return int(actions[int(np.argmax(visits))])

        if visits.sum() <= 0:
            return int(np.random.choice(actions))

        if self.config.temperature == 1.0:
            probs = visits / visits.sum()
        else:
            v = visits ** (1.0 / self.config.temperature)
            v_sum = v.sum()
            if v_sum <= 0:
                return int(np.random.choice(actions))
            probs = v / v_sum

        return int(np.random.choice(actions, p=probs))
