"""Image drag-and-drop zone with thumbnail preview for medical images."""

import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.utils import SUPPORTED_IMAGE_EXTENSIONS
from i18n import t


class ImageDropZone(QWidget):
    """Drop zone for medical images with thumbnail preview."""

    file_selected = pyqtSignal(str)
    file_removed = pyqtSignal()

    def __init__(
        self,
        accepted_extensions: Optional[List[str]] = None,
        placeholder_text: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._accepted_extensions = accepted_extensions or list(SUPPORTED_IMAGE_EXTENSIONS)
        self._placeholder_text = placeholder_text or t("xray.drop_text")
        self._current_file = ""
        self._drag_over = False
        self.setAcceptDrops(True)
        self.setObjectName("imageDropZone")
        self.setMinimumHeight(180)
        self._setup_ui()

    def _setup_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon_label = QLabel("\U0001f4f7")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setProperty("class", "dropZoneIcon")

        self._text_label = QLabel(self._placeholder_text)
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_label.setWordWrap(True)
        self._text_label.setProperty("class", "dropZoneText")

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setFixedHeight(140)
        self._preview_label.hide()

        self._file_info_label = QLabel()
        self._file_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._file_info_label.setProperty("class", "dropZoneFileInfo")
        self._file_info_label.hide()

        self._remove_row = QWidget()
        remove_layout = QHBoxLayout(self._remove_row)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._remove_btn = QPushButton(t("common.cancel"))
        self._remove_btn.setProperty("class", "removeFileButton")
        self._remove_btn.setFixedWidth(100)
        self._remove_btn.clicked.connect(self.reset)
        remove_layout.addWidget(self._remove_btn)
        self._remove_row.hide()

        self._layout.addWidget(self._icon_label)
        self._layout.addWidget(self._text_label)
        self._layout.addWidget(self._preview_label)
        self._layout.addWidget(self._file_info_label)
        self._layout.addWidget(self._remove_row)

    def _validate_extension(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self._accepted_extensions

    def _set_file(self, file_path: str):
        self._current_file = file_path
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        from core.utils import format_file_size
        self._file_info_label.setText(f"{filename} ({format_file_size(file_size)})")
        self._file_info_label.show()

        # Show thumbnail preview
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                200, 140,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)
            self._preview_label.show()
        else:
            self._preview_label.setText(filename)
            self._preview_label.show()

        self._icon_label.hide()
        self._text_label.hide()
        self._remove_row.show()

        self.file_selected.emit(file_path)

    def _browse_file(self):
        ext_filter = " ".join(f"*{e}" for e in self._accepted_extensions)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            t("common.browse"),
            "",
            f"Medical Images ({ext_filter})",
        )
        if file_path and self._validate_extension(file_path):
            self._set_file(file_path)

    def reset(self):
        """Remove the current file and reset to placeholder state."""
        self._current_file = ""
        self._preview_label.hide()
        self._preview_label.clear()
        self._file_info_label.hide()
        self._remove_row.hide()
        self._icon_label.show()
        self._text_label.show()
        self._drag_over = False
        self.update()
        self.file_removed.emit()

    def get_file_path(self) -> str:
        return self._current_file

    # --- Drag and drop ---

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self._validate_extension(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self._drag_over = True
                self.update()

    def dragLeaveEvent(self, event):
        self._drag_over = False
        self.update()

    def dropEvent(self, event: QDropEvent):
        self._drag_over = False
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self._validate_extension(file_path):
                self._set_file(file_path)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._current_file:
            self._browse_file()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._drag_over:
            pen = QPen(QColor("#007AFF"), 2, Qt.PenStyle.DashLine)
        elif self._current_file:
            pen = QPen(QColor("#34C759"), 2, Qt.PenStyle.SolidLine)
        else:
            pen = QPen(QColor("#888888"), 2, Qt.PenStyle.DashLine)

        pen.setDashPattern([8, 4])
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        painter.end()
