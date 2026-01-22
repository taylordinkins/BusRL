"""GUI package for the Bus board game.

This package provides a PySide6 based graphical interface for playing Bus.
It implements the GameRenderer and ActionPrompter interfaces from the driver module.

Main components:
- MainWindow: The main application window
- BoardWidget: Interactive board graph visualization
- ActionBoardWidget: Action board with marker placement display
- PlayerInfoWidget: Player status and resources
- GameInfoWidget: Global game state (time clock, round, etc.)
- GUIRenderer: Implementation of GameRenderer for the GUI
- GUIPrompter: Implementation of ActionPrompter for the GUI
- GameController: Integrates game engine with GUI event loop

Usage:
    python -m gui.app
"""

from gui.main_window import MainWindow
from gui.widgets import BoardWidget, ActionBoardWidget, PlayerInfoWidget, GameInfoWidget
from gui.gui_renderer import GUIRenderer
from gui.gui_prompter import GUIPrompter
from gui.game_controller import GameController
from gui.app import BusApp, main

__all__ = [
    "MainWindow",
    "BoardWidget",
    "ActionBoardWidget",
    "PlayerInfoWidget",
    "GameInfoWidget",
    "GUIRenderer",
    "GUIPrompter",
    "GameController",
    "BusApp",
    "main",
]
