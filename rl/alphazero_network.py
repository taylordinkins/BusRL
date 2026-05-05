"""AlphaZero-style neural network for Bus game.

Standalone network with a shared trunk MLP, a policy head (flat or
per-phase), and a value head. Not tied to SB3/MaskablePPO — saved and
loaded as a plain .pt state dict plus a _config.json sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlphaZeroNetworkConfig:
    """Configuration for AlphaZeroNetwork."""

    obs_dim: int
    # Number of actions in the flat action space (= BusEnv._max_head_actions).
    # The plan refers to this as "1670" but the actual value is board-dependent.
    num_actions: int
    num_players: int = 4
    trunk_layers: list = field(default_factory=lambda: [512, 512, 256])
    use_per_phase_heads: bool = False
    trunk_activation: str = "relu"
    use_layer_norm: bool = True


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class AlphaZeroNetwork(nn.Module):
    """Neural network with shared trunk, policy head(s), and value head.

    Policy head modes (controlled by config.use_per_phase_heads):
      Flat:      one Linear(trunk_dim, num_actions) for all phases
      Per-phase: 10 Linear layers, one per HeadId, sized by head_catalog_sizes

    Value head: Linear → activation → Linear(trunk_dim, num_players) → Sigmoid
                Output shape: (batch, num_players), each entry in [0, 1]
    """

    def __init__(
        self,
        config: AlphaZeroNetworkConfig,
        head_catalog_sizes: Optional[dict[int, int]] = None,
    ):
        """
        Args:
            config: Network hyperparameters.
            head_catalog_sizes: Required when use_per_phase_heads=True.
                Maps HeadId int value → number of actions for that head.
        """
        super().__init__()
        self.config = config

        # ── Trunk ────────────────────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = config.obs_dim
        for out_dim in config.trunk_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(_make_activation(config.trunk_activation))
            in_dim = out_dim
        self.trunk = nn.Sequential(*layers)
        trunk_out = in_dim

        # ── Policy head(s) ───────────────────────────────────────────────────
        if config.use_per_phase_heads:
            if head_catalog_sizes is None:
                raise ValueError(
                    "head_catalog_sizes is required when use_per_phase_heads=True"
                )
            self.policy_heads: Optional[nn.ModuleList] = nn.ModuleList(
                [nn.Linear(trunk_out, head_catalog_sizes[i]) for i in range(10)]
            )
            self.policy_head: Optional[nn.Linear] = None
        else:
            self.policy_head = nn.Linear(trunk_out, config.num_actions)
            self.policy_heads = None

        # ── Value head ───────────────────────────────────────────────────────
        mid = max(trunk_out // 2, config.num_players)
        self.value_head = nn.Sequential(
            nn.Linear(trunk_out, mid),
            _make_activation(config.trunk_activation),
            nn.Linear(mid, config.num_players),
            nn.Sigmoid(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        head_id: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs:     Shape (batch, obs_dim) or (obs_dim,).
            head_id: Active head ID int (required for per-phase mode).
            mask:    Boolean action mask; True = valid. Applied with -1e9 fill.

        Returns:
            policy_logits: (batch, num_actions) or (batch, head_size[head_id])
            value:         (batch, num_players)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self.trunk(obs)
        value = self.value_head(features)

        if self.config.use_per_phase_heads:
            if head_id is None:
                raise ValueError("head_id is required for per-phase policy heads")
            logits = self.policy_heads[head_id](features)
        else:
            logits = self.policy_head(features)

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            # In per-phase mode, env masks can be max_head_actions wide while
            # the active head logits are head-local. Trim to match logits.
            if mask.size(-1) != logits.size(-1):
                if mask.size(-1) > logits.size(-1):
                    mask = mask[..., : logits.size(-1)]
                else:
                    raise RuntimeError(
                        f"Mask width {mask.size(-1)} is smaller than logits width {logits.size(-1)}"
                    )
            logits = logits.masked_fill(~mask, -1e9)

        return logits, value

    # ── Convenience inference methods ────────────────────────────────────────

    def get_policy_priors(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        head_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Softmax action probabilities for a single observation.

        Returns:
            priors: np.ndarray, shape (num_actions,) or (head_size[head_id],)
        """
        if device is None:
            device = next(self.parameters()).device

        local_mask = np.asarray(mask, dtype=np.bool_)
        if (
            self.config.use_per_phase_heads
            and head_id is not None
            and self.policy_heads is not None
        ):
            head_size = self.policy_heads[head_id].out_features
            if local_mask.shape[0] != head_size:
                local_mask = local_mask[:head_size]

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(local_mask, dtype=torch.bool, device=device)
            logits, _ = self.forward(obs_t, head_id=head_id, mask=mask_t)
            priors = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        if np.isnan(priors).any():
            valid = np.where(local_mask)[0]
            priors = np.zeros_like(priors, dtype=np.float32)
            if len(valid) > 0:
                priors[valid] = 1.0 / len(valid)

        return priors

    def get_value(
        self,
        obs: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Value estimates for a single observation.

        Returns:
            values: np.ndarray, shape (num_players,), each in [0, 1]
        """
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            features = self.trunk(obs_t)
            value = self.value_head(features).squeeze(0)
            return value.cpu().numpy()

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save to {path}.pt + {path}_config.json."""
        path = Path(path)
        torch.save(self.state_dict(), path)
        cfg = {
            "obs_dim": int(self.config.obs_dim),
            "num_actions": int(self.config.num_actions),
            "num_players": int(self.config.num_players),
            "trunk_layers": [int(x) for x in self.config.trunk_layers],
            "use_per_phase_heads": bool(self.config.use_per_phase_heads),
            "trunk_activation": self.config.trunk_activation,
            "use_layer_norm": bool(self.config.use_layer_norm),
        }
        if self.config.use_per_phase_heads:
            cfg["head_catalog_sizes"] = {
                i: int(self.policy_heads[i].out_features) for i in range(10)
            }
        config_path = path.parent / (path.stem + "_config.json")
        config_path.write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AlphaZeroNetwork":
        """Load from {path}.pt + {path}_config.json."""
        path = Path(path)
        config_path = path.parent / (path.stem + "_config.json")
        data = json.loads(config_path.read_text())

        head_catalog_sizes = None
        if data.get("use_per_phase_heads"):
            head_catalog_sizes = {
                int(k): v for k, v in data["head_catalog_sizes"].items()
            }

        config = AlphaZeroNetworkConfig(
            obs_dim=data["obs_dim"],
            num_actions=data["num_actions"],
            num_players=data["num_players"],
            trunk_layers=data["trunk_layers"],
            use_per_phase_heads=data["use_per_phase_heads"],
            trunk_activation=data["trunk_activation"],
            use_layer_norm=data["use_layer_norm"],
        )
        net = cls(config, head_catalog_sizes=head_catalog_sizes)
        net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return net
