"""About dialog showing application info and credits."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from i18n import t


class AboutDialog(QDialog):
    """About LocalMedScan dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(t("about.title"))
        self.setFixedSize(400, 320)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel("\U0001fa7a")  # Stethoscope emoji
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(t("app.title"))
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px;")

        version = QLabel(t("about.version", version="1.0.0"))
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("color: #888;")

        desc = QLabel(t("about.description"))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)

        org = QLabel(t("about.org"))
        org.setAlignment(Qt.AlignmentFlag.AlignCenter)
        org.setStyleSheet("font-weight: bold;")

        mission = QLabel(t("about.mission"))
        mission.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mission.setStyleSheet("font-style: italic; color: #888;")
        mission.setWordWrap(True)

        license_label = QLabel(t("about.license"))
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_label.setStyleSheet("font-size: 11px; color: #888;")

        close_btn = QPushButton(t("about.close"))
        close_btn.setProperty("class", "secondaryButton")
        close_btn.clicked.connect(self.accept)

        layout.addWidget(icon_label)
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addWidget(desc)
        layout.addWidget(org)
        layout.addWidget(mission)
        layout.addWidget(license_label)
        layout.addStretch()
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
