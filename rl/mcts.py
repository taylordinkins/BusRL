"""Monte Carlo Tree Search with policy/value guidance.

This module provides MCTS implementation that uses a trained policy network
as prior and value network for leaf evaluation. Designed for use with
MaskablePPO models at inference time.

Key features:
- PUCT-based tree traversal (policy + UCB)
- Action masking for legal moves only
- Value network leaf evaluation
- Optional Dirichlet noise at root for exploration
- Temperature-based action selection
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from .bus_env import BusEnv


@dataclass
class MCTSConfig:
    """Configuration for MCTS search.

    Attributes:
        n_simulations: Number of MCTS simulations per search.
        c_puct: Exploration constant for PUCT formula.
        temperature: Temperature for final action selection (0 = greedy).
        use_value_network: If True, use value network for leaf eval. If False, use rollout.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_epsilon: Weight of Dirichlet noise (0 = no noise).
        max_rollout_depth: Maximum depth for random rollouts (if not using value network).
    """
    n_simulations: int = 100
    c_puct: float = 1.5
    temperature: float = 1.0
    use_value_network: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_rollout_depth: int = 50


class MCTSNode:
    """A node in the MCTS tree.

    Each node represents a game state, with edges to children representing
    actions. Stores visit counts, value estimates, and policy priors.
    """

    def __init__(
        self,
        env: "BusEnv",
        parent: Optional["MCTSNode"] = None,
        action_idx: int = -1,
        prior: float = 0.0,
    ):
        """Initialize MCTS node.

        Args:
            env: Cloned environment at this state.
            parent: Parent node (None for root).
            action_idx: Action that led to this node (-1 for root).
            prior: Policy prior probability for this action.
        """
        self.env = env
        self.parent = parent
        self.action_idx = action_idx
        self.prior = prior

        # Statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0

        # Children: action_idx -> MCTSNode
        self.children: dict[int, MCTSNode] = {}

        # Expansion state
        self.is_expanded: bool = False
        self.is_terminal: bool = False

        # Cache valid actions
        self._valid_actions: Optional[np.ndarray] = None

    @property
    def q_value(self) -> float:
        """Average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def valid_actions(self) -> np.ndarray:
        """Get array of valid action indices."""
        if self._valid_actions is None:
            mask = self.env.action_masks()
            self._valid_actions = np.where(mask)[0]
        return self._valid_actions

    def ucb_score(self, c_puct: float) -> float:
        """Calculate UCB score with policy prior (PUCT formula).

        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)

        Args:
            c_puct: Exploration constant.

        Returns:
            UCB score for this node.
        """
        if self.parent is None:
            return 0.0

        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        return self.q_value + exploration

    def select_child(self, c_puct: float) -> "MCTSNode":
        """Select best child using UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Child node with highest UCB score.
        """
        best_score = float("-inf")
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, priors: np.ndarray) -> None:
        """Expand node by creating children for all valid actions.

        Args:
            priors: Policy prior probabilities for all actions.
        """
        if self.is_expanded or self.is_terminal:
            return

        mask = self.env.action_masks()

        for action_idx in self.valid_actions:
            # Clone environment and take action
            child_env = self.env.clone()
            obs, reward, terminated, truncated, info = child_env.step(int(action_idx))

            child = MCTSNode(
                env=child_env,
                parent=self,
                action_idx=int(action_idx),
                prior=float(priors[action_idx]) if mask[action_idx] else 0.0,
            )

            if terminated or truncated:
                child.is_terminal = True

            self.children[int(action_idx)] = child

        self.is_expanded = True

    def backpropagate(self, value: float) -> None:
        """Backpropagate value through the tree to root.

        Args:
            value: Value estimate to propagate.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # Flip value for opponent's perspective in alternating games
            # For Bus (shared policy), we keep the same sign
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search with policy/value network guidance.

    Uses a trained MaskablePPO model to:
    1. Provide action priors for tree expansion
    2. Evaluate leaf nodes using value network

    Example:
        >>> model = MaskablePPO.load("bus_model.zip")
        >>> mcts = MCTS(model, config=MCTSConfig(n_simulations=100))
        >>> action = mcts.search(env)
    """

    def __init__(
        self,
        policy_model: "MaskablePPO",
        config: Optional[MCTSConfig] = None,
    ):
        """Initialize MCTS.

        Args:
            policy_model: Trained MaskablePPO model for priors and values.
            config: MCTS configuration. Uses defaults if None.
        """
        self.model = policy_model
        self.config = config or MCTSConfig()
        self.device = next(policy_model.policy.parameters()).device

    def search(self, env: "BusEnv") -> int:
        """Run MCTS search from current state.

        Args:
            env: Current environment state.

        Returns:
            Best action index based on visit counts.
        """
        # Create root node
        root = MCTSNode(env=env.clone())

        # Get initial priors and expand root
        obs = env._get_observation()
        mask = env.action_masks()
        priors = self._get_policy_priors(obs, mask)

        # Add Dirichlet noise at root for exploration
        if self.config.dirichlet_epsilon > 0:
            priors = self._add_dirichlet_noise(priors, mask)

        root.expand(priors)

        # Run simulations
        for _ in range(self.config.n_simulations):
            node = root

            # Selection: traverse to leaf
            while node.is_expanded and not node.is_terminal:
                node = node.select_child(self.config.c_puct)

            # Expansion and evaluation
            if not node.is_terminal:
                # Get observation and mask for this node
                node_obs = node.env._get_observation()
                node_mask = node.env.action_masks()

                # Evaluate leaf
                if self.config.use_value_network:
                    value = self._get_value_estimate(node_obs)
                else:
                    value = self._rollout(node.env)

                # Expand if not at terminal and has valid actions
                if np.any(node_mask):
                    node_priors = self._get_policy_priors(node_obs, node_mask)
                    node.expand(node_priors)
            else:
                # Terminal node - use actual game outcome via reward calculator for consistent scaling
                try:
                    reward_calculator = node.env.get_wrapper_attr("_reward_calculator")
                    current_player = env.get_current_player()
                    value = reward_calculator.compute_reward(
                        state=node.env.unwrapped._engine.state,
                        prev_state=node.env.unwrapped._prev_state,
                        player_id=current_player,
                        done=True
                    )
                except (AttributeError, Exception):
                    # Fallback to raw scores if calculator not accessible
                    scores = node.env._get_info().get("scores", {})
                    if scores:
                        current_player = env.get_current_player()
                        max_score = max(scores.values())
                        my_score = scores.get(current_player, 0)
                        value = float(my_score - max_score)
                    else:
                        value = 0.0

            # Backpropagation
            node.backpropagate(value)

        # Select action based on visit counts
        return self._select_action(root)

    def search_with_stats(self, env: "BusEnv") -> tuple[int, dict]:
        """Run MCTS search and return statistics.

        Args:
            env: Current environment state.

        Returns:
            Tuple of (best_action, stats_dict).
        """
        # Create root node
        root = MCTSNode(env=env.clone())

        # Get initial priors and expand root
        obs = env._get_observation()
        mask = env.action_masks()
        priors = self._get_policy_priors(obs, mask)

        if self.config.dirichlet_epsilon > 0:
            priors = self._add_dirichlet_noise(priors, mask)

        root.expand(priors)

        # Run simulations
        for _ in range(self.config.n_simulations):
            node = root

            while node.is_expanded and not node.is_terminal:
                node = node.select_child(self.config.c_puct)

            if not node.is_terminal:
                node_obs = node.env._get_observation()
                node_mask = node.env.action_masks()

                if self.config.use_value_network:
                    value = self._get_value_estimate(node_obs)
                else:
                    value = self._rollout(node.env)

                if np.any(node_mask):
                    node_priors = self._get_policy_priors(node_obs, node_mask)
                    node.expand(node_priors)
            else:
                # Terminal node - use actual game outcome via reward calculator for consistent scaling
                try:
                    reward_calculator = node.env.get_wrapper_attr("_reward_calculator")
                    current_player = env.get_current_player()
                    value = reward_calculator.compute_reward(
                        state=node.env.unwrapped._engine.state,
                        prev_state=node.env.unwrapped._prev_state,
                        player_id=current_player,
                        done=True
                    )
                except (AttributeError, Exception):
                    # Fallback to raw scores if calculator not accessible
                    scores = node.env._get_info().get("scores", {})
                    if scores:
                        current_player = env.get_current_player()
                        max_score = max(scores.values())
                        my_score = scores.get(current_player, 0)
                        value = float(my_score - max_score)
                    else:
                        value = 0.0

            node.backpropagate(value)

        action = self._select_action(root)

        # Collect statistics
        stats = {
            "root_value": float(root.q_value),
            "root_visits": int(root.visit_count),
            "child_visits": {int(a): int(c.visit_count) for a, c in root.children.items()},
            "child_values": {int(a): float(c.q_value) for a, c in root.children.items()},
            "selected_action": int(action),
            "selected_visits": int(root.children.get(action, root).visit_count),
        }

        return action, stats

    def _get_policy_priors(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy network.

        Args:
            obs: Observation array.
            mask: Boolean action mask.

        Returns:
            Probability distribution over actions.
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            # Get features through the policy network
            features = self.model.policy.extract_features(obs_tensor)

            # Handle both shared and separate feature extractors
            if self.model.policy.share_features_extractor:
                latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            else:
                pi_features = self.model.policy.pi_features_extractor(obs_tensor)
                latent_pi = self.model.policy.mlp_extractor.forward_actor(pi_features)

            # Get action logits
            action_logits = self.model.policy.action_net(latent_pi)

            # Apply mask: set invalid actions to large negative value
            masked_logits = action_logits.clone()
            masked_logits[~mask_tensor] = float("-inf")

            # Softmax to get probabilities
            priors = F.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()

            # Handle NaN from all-masked case
            if np.isnan(priors).any():
                priors = np.zeros_like(priors)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    priors[valid_indices] = 1.0 / len(valid_indices)

            return priors

    def _get_value_estimate(self, obs: np.ndarray) -> float:
        """Get value estimate from the policy's value head.

        Args:
            obs: Observation array.

        Returns:
            Value estimate in [-1, 1] range (approximately).
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Get features
            features = self.model.policy.extract_features(obs_tensor)

            # Handle both shared and separate feature extractors
            if self.model.policy.share_features_extractor:
                latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
            else:
                vf_features = self.model.policy.vf_features_extractor(obs_tensor)
                latent_vf = self.model.policy.mlp_extractor.forward_critic(vf_features)

            # Get value
            value = self.model.policy.value_net(latent_vf)

            return float(value.squeeze().cpu().numpy())

    def _rollout(self, env: "BusEnv") -> float:
        """Perform random rollout from state for value estimation.

        Args:
            env: Environment to roll out from.

        Returns:
            Value estimate based on rollout outcome.
        """
        rollout_env = env.clone()
        current_player = env.get_current_player()

        for _ in range(self.config.max_rollout_depth):
            if rollout_env._engine.is_game_over():
                break

            mask = rollout_env.action_masks()
            valid_actions = np.where(mask)[0]

            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = rollout_env.step(int(action))

            if terminated or truncated:
                break

        # Get final scores via reward calculator
        try:
            reward_calculator = rollout_env.get_wrapper_attr("_reward_calculator")
            return float(reward_calculator.compute_reward(
                state=rollout_env.unwrapped._engine.state,
                prev_state=rollout_env.unwrapped._prev_state,
                player_id=current_player,
                done=True
            ))
        except (AttributeError, Exception):
            scores = rollout_env._get_info().get("scores", {})
            if scores:
                max_score = max(scores.values())
                my_score = scores.get(current_player, 0)
                return float(my_score - max_score)
        return 0.0

    def _add_dirichlet_noise(self, priors: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add Dirichlet noise to root priors for exploration.

        Args:
            priors: Original policy priors.
            mask: Valid action mask.

        Returns:
            Priors with Dirichlet noise added.
        """
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return priors

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_indices))

        noisy_priors = priors.copy()
        for i, idx in enumerate(valid_indices):
            noisy_priors[idx] = (
                (1 - self.config.dirichlet_epsilon) * priors[idx]
                + self.config.dirichlet_epsilon * noise[i]
            )

        # Renormalize
        total = noisy_priors[valid_indices].sum()
        if total > 0:
            noisy_priors[valid_indices] /= total

        return noisy_priors

    def _select_action(self, root: MCTSNode) -> int:
        """Select action from root based on visit counts.

        Args:
            root: Root node after search.

        Returns:
            Selected action index.
        """
        if not root.children:
            # Fallback to random valid action
            valid_actions = root.valid_actions
            return int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0

        if self.config.temperature == 0:
            # Greedy: select most visited
            return max(root.children.keys(), key=lambda a: root.children[a].visit_count)

        # Temperature-based selection
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

        if self.config.temperature == 1.0:
            # Proportional to visits
            probs = visits / visits.sum()
        else:
            # Apply temperature
            visits_temp = visits ** (1.0 / self.config.temperature)
            probs = visits_temp / visits_temp.sum()

        return int(np.random.choice(actions, p=probs))
