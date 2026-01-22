"""Main entry point for the Bus GUI application."""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.main_window import MainWindow
from gui.game_controller import GameController


class BusApp:
    """Main application class for the Bus GUI."""

    def __init__(self):
        """Initialize the application."""
        self._app: Optional[QApplication] = None
        self._window: Optional[MainWindow] = None
        self._controller: Optional[GameController] = None

    def run(self) -> int:
        """Run the application.

        Returns:
            Exit code (0 for success).
        """
        # Create Qt application
        self._app = QApplication(sys.argv)
        self._app.setApplicationName("Bus")
        self._app.setApplicationDisplayName("Bus - Digital Board Game")

        # Set application style
        self._app.setStyle("Fusion")

        # Create main window
        self._window = MainWindow()

        # Create game controller
        self._controller = GameController(self._window)

        # Show window
        self._window.show()

        # Run event loop
        return self._app.exec()


def main() -> int:
    """Main entry point."""
    app = BusApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
