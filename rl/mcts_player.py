"""AlphaZero player wrappers for GUI and evaluation use.

Full rewrite of the inference-only MCTSPlayer / PolicyPlayer that were
coupled to MaskablePPO. Both new classes expose the same
``get_action(env) -> int`` interface so they are drop-in replacements
in the GUI (gui/game_controller.py, gui/dialogs.py).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .alphazero_network import AlphaZeroNetwork
    from .mcts import MCTSConfig
    from .bus_env import BusEnv


class AlphaZeroPlayer:
    """Plays using AlphaZeroNetwork + MCTS search (full strength).

    Use n_simulations=0 (or mcts_config with n_simulations=0) to fall
    back to policy-only mode without any tree search.
    """

    def __init__(
        self,
        network: "AlphaZeroNetwork",
        mcts_config: Optional["MCTSConfig"] = None,
        *,
        n_simulations: int = 400,
        c_puct: float = 1.5,
        dirichlet_epsilon: float = 0.0,   # no noise for evaluation
    ):
        """
        Args:
            network:          Loaded AlphaZeroNetwork.
            mcts_config:      Full MCTSConfig (overrides individual params).
            n_simulations:    MCTS simulations per move (ignored if mcts_config given).
            c_puct:           PUCT exploration constant.
            dirichlet_epsilon: Noise weight at root (0 = deterministic eval).
        """
        from .mcts import AlphaZeroMCTS, MCTSConfig

        self.network = network

        if mcts_config is None:
            mcts_config = MCTSConfig(
                n_simulations=n_simulations,
                c_puct=c_puct,
                temperature=0.0,
                dirichlet_epsilon=dirichlet_epsilon,
                num_players=network.config.num_players,
            )

        self.mcts = AlphaZeroMCTS(network, mcts_config)
        self._policy_player = AlphaZeroPolicyPlayer(network)

    def get_action(self, env: "BusEnv") -> int:
        """Select action using MCTS search."""
        if self.mcts.config.n_simulations <= 0:
            return self._policy_player.get_action(env)
        return self.mcts.search(env)

    def get_action_with_stats(self, env: "BusEnv") -> tuple[int, dict]:
        """Select action and return per-child visit counts for debugging."""
        from .mcts import AlphaZeroMCTS

        if self.mcts.config.n_simulations <= 0:
            action = self._policy_player.get_action(env)
            stats = {
                "mode": "policy_only",
                "root_visits": 0,
                "root_value": None,
                "child_visits": {},
                "child_values": {},
                "selected_action": int(action),
            }
            return action, stats

        root = self.mcts._build_root(env, add_noise=(self.mcts.config.dirichlet_epsilon > 0))
        for _ in range(self.mcts.config.n_simulations):
            self.mcts._simulate(root)

        action = self.mcts._select_action(
            root, move_number=self.mcts.config.temperature_threshold + 1
        )

        stats = {
            "root_visits": root.visit_count,
            "root_value": float(root.value_sum[root.current_player] / max(root.visit_count, 1)),
            "child_visits": {a: c.visit_count for a, c in root.children.items()},
            "child_values": {
                a: float(c.value_sum[root.current_player] / max(c.visit_count, 1))
                for a, c in root.children.items()
            },
            "selected_action": int(action),
        }
        return action, stats


class AlphaZeroPolicyPlayer:
    """Direct policy inference without any MCTS tree search.

    Faster than AlphaZeroPlayer; useful as a baseline or for rapid
    evaluation where per-move latency matters.
    """

    def __init__(self, network: "AlphaZeroNetwork"):
        """
        Args:
            network: Loaded AlphaZeroNetwork.
        """
        self.network = network
        self._device = next(network.parameters()).device

    def get_action(self, env: "BusEnv") -> int:
        """Select action greedily from the network's policy head."""
        import torch
        import torch.nn.functional as F

        obs = env._get_observation()
        mask = env.action_masks()
        decision = env._get_decision_context()
        head_id_obj = decision.get("head_id") if decision else None
        head_id = head_id_obj.value if head_id_obj is not None else None

        self.network.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device)
            logits, _ = self.network(obs_t, head_id=head_id, mask=mask_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if np.isnan(probs).any():
            valid = np.where(mask)[0]
            return int(np.random.choice(valid)) if len(valid) > 0 else 0

        return int(np.argmax(probs))
