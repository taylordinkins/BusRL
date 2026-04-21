"""Game setup configuration dataclasses shared by the dialog and controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class PlayerConfig:
    player_id: int
    is_human: bool
    checkpoint_path: Optional[str] = None
    # Populated by GameController after the model is loaded; not set by dialog.
    policy_player: Optional[Any] = field(default=None, repr=False)


@dataclass
class GameSetupConfig:
    player_configs: list[PlayerConfig]

    @property
    def num_players(self) -> int:
        return len(self.player_configs)

    @property
    def is_spectate_mode(self) -> bool:
        return all(not pc.is_human for pc in self.player_configs)
