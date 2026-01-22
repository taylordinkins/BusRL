"""Game engine for the Bus board game.

This module provides the game logic including:
- Phase state machine for game flow control
- Action resolvers for each action area
- Game engine for coordinating game play
"""

from .phase_machine import (
    PhaseMachine,
    PhaseTransitionResult,
    PHASE_TRANSITIONS,
)

from .setup import (
    SetupManager,
    SetupAction,
    SetupValidationResult,
    initialize_game,
)

from .game_engine import (
    GameEngine,
    Action,
    ActionType,
    StepResult,
)

__all__ = [
    # Phase machine
    "PhaseMachine",
    "PhaseTransitionResult",
    "PHASE_TRANSITIONS",
    # Setup
    "SetupManager",
    "SetupAction",
    "SetupValidationResult",
    "initialize_game",
    # Game engine
    "GameEngine",
    "Action",
    "ActionType",
    "StepResult",
]
