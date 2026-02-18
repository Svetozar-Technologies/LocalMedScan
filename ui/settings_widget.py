"""Application settings tab — language, theme, model management."""

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.model_manager import ModelManager, get_model_manager
from core.utils import format_file_size
from i18n import LANGUAGES, t
from ui.theme import ThemeManager


class SettingsWidget(QWidget):
    """User-facing settings for language, theme, and model management."""

    def __init__(self, theme_manager: ThemeManager, parent=None):
        super().__init__(parent)
        self._theme_manager = theme_manager
        self._model_manager = get_model_manager()
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel(t("settings.title"))
        title.setProperty("class", "sectionTitle")
        subtitle = QLabel(t("settings.subtitle"))
        subtitle.setProperty("class", "sectionSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # --- Language section ---
        lang_header = QLabel(t("settings.language"))
        lang_header.setProperty("class", "sectionTitle")
        lang_header.setStyleSheet("font-size: 16px; margin-top: 12px;")
        layout.addWidget(lang_header)

        lang_row = QHBoxLayout()
        self._lang_combo = QComboBox()
        settings = QSettings("Svetozar Technologies", "LocalMedScan")
        current = settings.value("language", "en")
        for code, info in sorted(LANGUAGES.items()):
            self._lang_combo.addItem(f"{info['native_name']} ({info['name']})", code)
            if code == current:
                self._lang_combo.setCurrentIndex(self._lang_combo.count() - 1)
        self._lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch()
        layout.addLayout(lang_row)

        lang_note = QLabel(t("settings.language_restart"))
        lang_note.setStyleSheet("font-size: 11px; color: #888; font-style: italic;")
        layout.addWidget(lang_note)

        # --- Theme section ---
        theme_header = QLabel(t("settings.theme"))
        theme_header.setProperty("class", "sectionTitle")
        theme_header.setStyleSheet("font-size: 16px; margin-top: 12px;")
        layout.addWidget(theme_header)

        theme_row = QHBoxLayout()
        self._light_btn = QPushButton(t("settings.theme_light"))
        self._light_btn.setProperty("class", "secondaryButton")
        self._light_btn.clicked.connect(lambda: self._set_theme("light"))

        self._dark_btn = QPushButton(t("settings.theme_dark"))
        self._dark_btn.setProperty("class", "secondaryButton")
        self._dark_btn.clicked.connect(lambda: self._set_theme("dark"))

        theme_row.addWidget(self._light_btn)
        theme_row.addWidget(self._dark_btn)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        # --- Models section ---
        models_header = QLabel(t("settings.models"))
        models_header.setProperty("class", "sectionTitle")
        models_header.setStyleSheet("font-size: 16px; margin-top: 12px;")
        layout.addWidget(models_header)

        models_subtitle = QLabel(t("settings.models_subtitle"))
        models_subtitle.setProperty("class", "sectionSubtitle")
        layout.addWidget(models_subtitle)

        # Model cards
        self._models_container = QVBoxLayout()
        self._models_container.setSpacing(8)
        layout.addLayout(self._models_container)

        self._total_size_label = QLabel()
        self._total_size_label.setStyleSheet("font-size: 12px; color: #888; margin-top: 8px;")
        layout.addWidget(self._total_size_label)

        self._refresh_models()

        # --- About section ---
        about_header = QLabel(t("settings.about"))
        about_header.setProperty("class", "sectionTitle")
        about_header.setStyleSheet("font-size: 16px; margin-top: 12px;")
        layout.addWidget(about_header)

        about_text = QLabel(
            f"{t('about.description')}\n"
            f"{t('about.version', version='1.0.0')}\n"
            f"{t('about.org')}\n"
            f"{t('about.mission')}\n"
            f"{t('about.license')}"
        )
        about_text.setProperty("class", "sectionSubtitle")
        about_text.setWordWrap(True)
        layout.addWidget(about_text)

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _on_language_changed(self, index: int):
        code = self._lang_combo.itemData(index)
        settings = QSettings("Svetozar Technologies", "LocalMedScan")
        settings.setValue("language", code)

    def _set_theme(self, mode: str):
        self._theme_manager.set_theme(mode)

    def _refresh_models(self):
        # Clear existing
        while self._models_container.count():
            item = self._models_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for model in self._model_manager.get_registry():
            card = self._create_model_card(model)
            self._models_container.addWidget(card)

        total = self._model_manager.get_total_size_formatted()
        self._total_size_label.setText(t("settings.total_size", size=total))

    def _create_model_card(self, model) -> QWidget:
        card = QWidget()
        card.setObjectName("resultCard")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(12)

        info_col = QVBoxLayout()
        name_label = QLabel(model.display_name)
        name_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        desc_label = QLabel(model.description)
        desc_label.setStyleSheet("font-size: 11px; color: #888;")
        desc_label.setWordWrap(True)

        size_str = format_file_size(int(model.size_mb * 1024 * 1024))
        meta_label = QLabel(f"{t('settings.model_size', size=size_str)} | Accuracy: {model.accuracy}")
        meta_label.setStyleSheet("font-size: 11px; color: #666;")

        info_col.addWidget(name_label)
        info_col.addWidget(desc_label)
        info_col.addWidget(meta_label)

        available = self._model_manager.is_model_available(model.name)
        status_label = QLabel(
            t("settings.model_downloaded") if available else t("settings.model_not_downloaded")
        )
        status_label.setStyleSheet(
            "font-size: 11px; color: #22C55E;" if available else "font-size: 11px; color: #F59E0B;"
        )

        row.addLayout(info_col, 1)
        row.addWidget(status_label)

        return card

    def refresh(self):
        """Reload settings state."""
        self._refresh_models()

    def cleanup(self):
        pass
