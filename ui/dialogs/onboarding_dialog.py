"""First-run onboarding dialog with medical disclaimer acceptance."""

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from i18n import t


class OnboardingDialog(QDialog):
    """Multi-page onboarding wizard. User MUST accept the medical disclaimer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("onboarding.welcome_title"))
        self.setMinimumSize(560, 440)
        self.setModal(True)
        self._accepted = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(16)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._create_welcome_page())
        self._stack.addWidget(self._create_privacy_page())
        self._stack.addWidget(self._create_disclaimer_page())
        self._stack.addWidget(self._create_model_page())

        layout.addWidget(self._stack, 1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(12)

        self._back_btn = QPushButton(t("onboarding.back"))
        self._back_btn.setFixedWidth(100)
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.hide()

        nav_row.addWidget(self._back_btn)
        nav_row.addStretch()

        self._next_btn = QPushButton(t("onboarding.next"))
        self._next_btn.setObjectName("primaryButton")
        self._next_btn.setFixedWidth(140)
        self._next_btn.clicked.connect(self._go_next)

        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        self._page_label = QLabel("1 / 4")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setProperty("class", "pageIndicator")
        layout.addWidget(self._page_label)

    def _create_welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        title = QLabel(t("onboarding.welcome_title"))
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("\U0001fa7a")
        icon.setStyleSheet("font-size: 64px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel(t("onboarding.welcome_text"))
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setProperty("class", "onboardingText")

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addStretch()
        return page

    def _create_privacy_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        title = QLabel(t("onboarding.privacy_title"))
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("\U0001f512")
        icon.setStyleSheet("font-size: 64px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel(t("onboarding.privacy_text"))
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setProperty("class", "onboardingText")

        features = QLabel(
            "\u2022 100% local processing — nothing leaves your device\n"
            "\u2022 No cloud uploads, no servers, no accounts\n"
            "\u2022 No telemetry or analytics\n"
            "\u2022 Open source — fully auditable code\n"
            "\u2022 MIT License — free forever"
        )
        features.setWordWrap(True)
        features.setAlignment(Qt.AlignmentFlag.AlignCenter)
        features.setProperty("class", "onboardingFeatures")

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(features)
        layout.addStretch()
        return page

    def _create_disclaimer_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        title = QLabel(t("onboarding.disclaimer_title"))
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("\u26a0\ufe0f")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        disclaimer_text = QLabel(t("disclaimer.full"))
        disclaimer_text.setWordWrap(True)
        disclaimer_text.setProperty("class", "disclaimerFullText")

        self._accept_checkbox = QCheckBox(t("disclaimer.accept_checkbox"))
        self._accept_checkbox.setProperty("class", "disclaimerCheckbox")
        self._accept_checkbox.stateChanged.connect(self._on_checkbox_changed)

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(disclaimer_text)
        layout.addSpacing(12)
        layout.addWidget(self._accept_checkbox)
        layout.addStretch()
        return page

    def _create_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        title = QLabel(t("onboarding.model_title"))
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("\U0001f9e0")
        icon.setStyleSheet("font-size: 64px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel(t("onboarding.model_text"))
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setProperty("class", "onboardingText")

        models_info = QLabel(
            "\u2022 Malaria Detection (MobileNetV2) — ~14 MB\n"
            "\u2022 TB/Pneumonia Screening (DenseNet121) — ~50 MB\n\n"
            "Models will download automatically on first use.\n"
            "After downloading, the app works completely offline."
        )
        models_info.setWordWrap(True)
        models_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        models_info.setProperty("class", "onboardingFeatures")

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(models_info)
        layout.addStretch()
        return page

    def _on_checkbox_changed(self, state):
        current_page = self._stack.currentIndex()
        if current_page == 2:
            self._next_btn.setEnabled(state == Qt.CheckState.Checked.value)

    def _go_next(self):
        current = self._stack.currentIndex()
        if current == self._stack.count() - 1:
            self._accepted = True
            settings = QSettings("Svetozar Technologies", "LocalMedScan")
            settings.setValue("onboarding_complete", True)
            self.accept()
        else:
            self._stack.setCurrentIndex(current + 1)
            self._update_nav()

    def _go_back(self):
        current = self._stack.currentIndex()
        if current > 0:
            self._stack.setCurrentIndex(current - 1)
            self._update_nav()

    def _update_nav(self):
        current = self._stack.currentIndex()
        total = self._stack.count()

        self._back_btn.setVisible(current > 0)
        self._page_label.setText(f"{current + 1} / {total}")

        if current == total - 1:
            self._next_btn.setText(t("onboarding.finish"))
        else:
            self._next_btn.setText(t("onboarding.next"))

        # Disable next on disclaimer page until checkbox is checked
        if current == 2:
            self._next_btn.setEnabled(self._accept_checkbox.isChecked())
        else:
            self._next_btn.setEnabled(True)

    def was_accepted(self) -> bool:
        return self._accepted

    @staticmethod
    def needs_onboarding() -> bool:
        settings = QSettings("Svetozar Technologies", "LocalMedScan")
        return not settings.value("onboarding_complete", False, type=bool)
