"""GUI prompter implementing the ActionPrompter interface."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PySide6.QtWidgets import QDialog

from engine.driver import ActionPrompter, ActionChoice

if TYPE_CHECKING:
    from gui.main_window import MainWindow


class GUIPrompter(ActionPrompter):
    """GUI implementation of ActionPrompter.

    This class presents choices to the user via dialog windows.
    """

    def __init__(self, main_window: 'MainWindow'):
        """Initialize the GUI prompter.

        Args:
            main_window: The main application window.
        """
        self._window = main_window

    def prompt_choice(
        self,
        message: str,
        choices: list[ActionChoice],
        allow_cancel: bool = False,
    ) -> Optional[ActionChoice]:
        """Prompt the player to make a choice.

        Args:
            message: The prompt message.
            choices: List of available choices.
            allow_cancel: Whether to allow canceling the choice.

        Returns:
            The selected choice, or None if canceled.
        """
        from gui.dialogs import ChoiceDialog

        dialog = ChoiceDialog(
            title="Make a Choice",
            message=message,
            choices=choices,
            allow_cancel=allow_cancel,
            parent=self._window
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_selected_choice()
        return None

    def prompt_confirmation(self, message: str) -> bool:
        """Prompt for yes/no confirmation.

        Args:
            message: The confirmation message.

        Returns:
            True if confirmed, False otherwise.
        """
        from gui.dialogs import ConfirmDialog

        dialog = ConfirmDialog(
            title="Confirm",
            message=message,
            parent=self._window
        )

        return dialog.exec() == QDialog.DialogCode.Accepted
