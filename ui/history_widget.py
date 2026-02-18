"""Screening history browser tab."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.history_db import get_history_db
from core.utils import HistoryEntry
from i18n import t


class _HistoryCard(QWidget):
    """Single history entry card."""

    def __init__(self, entry: HistoryEntry, on_delete=None, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._on_delete = on_delete
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("resultCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)

        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(64, 64)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self._entry.thumbnail:
            pixmap = QPixmap()
            pixmap.loadFromData(self._entry.thumbnail)
            if not pixmap.isNull():
                thumb_label.setPixmap(pixmap.scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            else:
                thumb_label.setText("\U0001f4f7")
        else:
            thumb_label.setText("\U0001f4f7")
            thumb_label.setStyleSheet("font-size: 28px; color: #888;")

        # Info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        type_label = QLabel(self._entry.screening_type.value.upper())
        type_label.setProperty("class", "sectionTitle")
        type_label.setStyleSheet("font-size: 14px;")

        conf_pct = f"{self._entry.overall_confidence * 100:.1f}%"
        detail = f"Confidence: {conf_pct} | Findings: {self._entry.findings_count} | Model: {self._entry.model_name}"
        detail_label = QLabel(detail)
        detail_label.setProperty("class", "sectionSubtitle")
        detail_label.setStyleSheet("font-size: 12px;")

        time_label = QLabel(self._entry.timestamp[:19].replace("T", " "))
        time_label.setStyleSheet("font-size: 11px; color: #999;")

        info_layout.addWidget(type_label)
        info_layout.addWidget(detail_label)
        info_layout.addWidget(time_label)

        # Delete button
        delete_btn = QPushButton(t("history.delete"))
        delete_btn.setProperty("class", "cancelButton")
        delete_btn.setFixedWidth(70)
        delete_btn.clicked.connect(self._delete)

        layout.addWidget(thumb_label)
        layout.addLayout(info_layout, 1)
        layout.addWidget(delete_btn)

    def _delete(self):
        db = get_history_db()
        db.delete(self._entry.id)
        if self._on_delete:
            self._on_delete()


class HistoryWidget(QWidget):
    """Browse past screening history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_history()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(32, 24, 32, 24)
        self._layout.setSpacing(16)

        # Header
        header_row = QHBoxLayout()
        title = QLabel(t("history.title"))
        title.setProperty("class", "sectionTitle")

        self._clear_btn = QPushButton(t("history.clear_all"))
        self._clear_btn.setProperty("class", "cancelButton")
        self._clear_btn.clicked.connect(self._clear_all)

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self._clear_btn)

        subtitle = QLabel(t("history.subtitle"))
        subtitle.setProperty("class", "sectionSubtitle")
        subtitle.setWordWrap(True)

        self._layout.addLayout(header_row)
        self._layout.addWidget(subtitle)

        # Content area for cards
        self._cards_layout = QVBoxLayout()
        self._cards_layout.setSpacing(8)
        self._layout.addLayout(self._cards_layout)
        self._layout.addStretch()

        scroll.setWidget(self._container)
        outer.addWidget(scroll)

    def _load_history(self):
        """Load and display history entries."""
        # Clear existing cards
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        db = get_history_db()
        entries = db.get_all(limit=100)

        if not entries:
            empty_label = QLabel(t("history.empty"))
            empty_label.setProperty("class", "historyEmpty")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._cards_layout.addWidget(empty_label)
            self._clear_btn.setEnabled(False)
        else:
            self._clear_btn.setEnabled(True)
            for entry in entries:
                card = _HistoryCard(entry, on_delete=self._load_history)
                self._cards_layout.addWidget(card)

    def refresh(self):
        """Reload history from database."""
        self._load_history()

    def _clear_all(self):
        reply = QMessageBox.question(
            self,
            t("common.warning"),
            t("history.clear_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            db = get_history_db()
            db.clear_all()
            self._load_history()

    def cleanup(self):
        pass
