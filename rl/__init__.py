"""Reinforcement Learning module for the Bus board game.

This module provides a Gymnasium-compatible environment for training
RL agents to play the Bus board game using PPO/MaskablePPO with
action masking.

Key components:
- BusEnv: Core Gymnasium environment
- MultiPolicyBusEnv: Wrapper for training against diverse opponents
- OpponentPool: Checkpoint management for self-play diversity
- EloTracker: Rating system for evaluating checkpoint performance
- MCTS: Monte Carlo Tree Search for enhanced inference
"""

from .config import (
    BoardConfig,
    ObservationConfig,
    ActionSpaceConfig,
    RewardConfig,
    DEFAULT_BOARD_CONFIG,
    DEFAULT_OBS_CONFIG,
    DEFAULT_ACTION_CONFIG,
    DEFAULT_REWARD_CONFIG,
)
from .observation import ObservationEncoder
from .action_space import ActionMapping
from .action_masking import ActionMaskGenerator
from .reward import RewardCalculator, StepRewardInfo
from .bus_env import BusEnv, make_bus_env
from .wrappers import BusEnvSelfPlayWrapper
from .opponent_pool import OpponentPool, CheckpointInfo, PoolConfig
from .elo_tracker import EloTracker, MatchResult, HeadToHeadStats
from .multi_policy_env import MultiPolicyBusEnv, PolicySlot, MatchRunner
from .mcts import MCTS, MCTSNode, MCTSConfig
from .mcts_player import MCTSPlayer, PolicyPlayer, MCTSPlayerStats
from .callbacks import (
    OpponentPoolCallback,
    OpponentPoolEvalCallback,
    MultiPolicyTrainingCallback,
)

__all__ = [
    # Configuration
    "BoardConfig",
    "ObservationConfig",
    "ActionSpaceConfig",
    "RewardConfig",
    "DEFAULT_BOARD_CONFIG",
    "DEFAULT_OBS_CONFIG",
    "DEFAULT_ACTION_CONFIG",
    "DEFAULT_REWARD_CONFIG",
    # Observation encoding
    "ObservationEncoder",
    # Action space
    "ActionMapping",
    # Action masking
    "ActionMaskGenerator",
    # Reward
    "RewardCalculator",
    "StepRewardInfo",
    # Environment
    "BusEnv",
    "make_bus_env",
    "BusEnvSelfPlayWrapper",
    # Multi-policy environment
    "MultiPolicyBusEnv",
    "PolicySlot",
    "MatchRunner",
    # Opponent Pool
    "OpponentPool",
    "CheckpointInfo",
    "PoolConfig",
    # Elo Rating System
    "EloTracker",
    "MatchResult",
    "HeadToHeadStats",
    # MCTS
    "MCTS",
    "MCTSNode",
    "MCTSConfig",
    "MCTSPlayer",
    "PolicyPlayer",
    "MCTSPlayerStats",
    # Callbacks
    "OpponentPoolCallback",
    "OpponentPoolEvalCallback",
    "MultiPolicyTrainingCallback",
]
