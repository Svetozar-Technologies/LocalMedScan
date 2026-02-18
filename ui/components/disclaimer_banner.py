"""Persistent medical disclaimer banner shown on all screening tabs."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


class DisclaimerBanner(QWidget):
    """Yellow/amber warning banner with medical disclaimer text.

    Always visible on screening tabs. Cannot be dismissed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("disclaimerBanner")
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        icon_label = QLabel("\u26a0")
        icon_label.setProperty("class", "disclaimerIcon")
        icon_label.setFixedWidth(24)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        from i18n import t
        text_label = QLabel(t("disclaimer.banner"))
        text_label.setProperty("class", "disclaimerText")
        text_label.setWordWrap(True)

        layout.addWidget(icon_label)
        layout.addWidget(text_label, 1)
