"""LocalMedScan — Offline AI Medical Image Screening Assistant.

Entry point for the desktop application.
Svetozar Technologies — AI Should Be Free. AI Should Be Private. AI Should Be Yours.
"""

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import i18n
from ui.theme import ThemeManager
from ui.dialogs.onboarding_dialog import OnboardingDialog


def main():
    """Application entry point."""
    # Handle PyInstaller frozen app
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))

    app = QApplication(sys.argv)
    app.setApplicationName("LocalMedScan")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Svetozar Technologies")

    # Initialize i18n before any UI
    i18n.init()
    if i18n.is_rtl():
        app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

    # Apply theme
    theme_manager = ThemeManager(app)
    theme_manager.apply_theme()

    # Show onboarding on first run
    if OnboardingDialog.needs_onboarding():
        dialog = OnboardingDialog()
        result = dialog.exec()
        if not dialog.was_accepted():
            sys.exit(0)

    # Import main window after onboarding (avoids loading heavy deps if user quits)
    from ui.main_window import MainWindow

    window = MainWindow(theme_manager)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
