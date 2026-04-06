"""Custom policies for Bus RL training."""

from __future__ import annotations

from typing import Any, Optional
import inspect

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class BusMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable policy with optional logit clamping and value/logit sanity checks."""

    def __init__(
        self,
        *args: Any,
        logit_clamp: bool = True,
        logit_clamp_min: float = -20.0,
        logit_clamp_max: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logit_clamp = bool(logit_clamp)
        self.logit_clamp_min = float(logit_clamp_min)
        self.logit_clamp_max = float(logit_clamp_max)
        if self.logit_clamp_min > self.logit_clamp_max:
            raise ValueError("logit_clamp_min must be <= logit_clamp_max")

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ):
        logits = self.action_net(latent_pi)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits before masking")
        if self.logit_clamp:
            if action_masks is not None:
                mask = (
                    action_masks
                    if isinstance(action_masks, torch.Tensor)
                    else torch.as_tensor(action_masks)
                ).to(dtype=torch.bool, device=logits.device)
                clamped = torch.clamp(logits, self.logit_clamp_min, self.logit_clamp_max)
                logits = torch.where(mask, clamped, logits)
            else:
                logits = torch.clamp(logits, self.logit_clamp_min, self.logit_clamp_max)
        proba_sig = inspect.signature(self.action_dist.proba_distribution)
        if "action_masks" in proba_sig.parameters:
            dist = self.action_dist.proba_distribution(
                action_logits=logits, action_masks=action_masks
            )
        else:
            dist = self.action_dist.proba_distribution(action_logits=logits)
            if action_masks is not None:
                mask = (
                    action_masks
                    if isinstance(action_masks, torch.Tensor)
                    else torch.as_tensor(action_masks)
                ).to(dtype=torch.bool, device=logits.device)
                if hasattr(self.action_dist, "apply_masking"):
                    self.action_dist.apply_masking(mask)
        masked_logits = dist.distribution.logits
        if not torch.isfinite(masked_logits).all():
            raise RuntimeError("Non-finite logits after masking/clamp")
        return dist

    def forward(self, obs, deterministic: bool = False, action_masks: Optional[torch.Tensor] = None):
        actions, values, log_prob = super().forward(
            obs, deterministic=deterministic, action_masks=action_masks
        )
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite value predictions")
        return actions, values, log_prob

    def evaluate_actions(
        self, obs, actions, action_masks: Optional[torch.Tensor] = None
    ):
        values, log_prob, entropy = super().evaluate_actions(
            obs, actions, action_masks=action_masks
        )
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite value predictions")
        return values, log_prob, entropy
