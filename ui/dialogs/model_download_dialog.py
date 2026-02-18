"""Model download dialog with progress."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from i18n import t
from workers.model_download_worker import ModelDownloadWorker


class ModelDownloadDialog(QDialog):
    """Dialog for downloading AI models with progress display."""

    def __init__(self, model_name: str, url: str, output_path: str, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._url = url
        self._output_path = output_path
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(t("settings.model_download"))
        self.setFixedWidth(420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._title_label = QLabel(f"Downloading {self._model_name}...")
        self._title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self._title_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Preparing download...")
        self._status_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self._status_label)

        self._cancel_btn = QPushButton(t("common.cancel"))
        self._cancel_btn.setProperty("class", "cancelButton")
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def start_download(self):
        """Begin the download."""
        self._worker = ModelDownloadWorker(
            self._url, self._output_path, parent=self
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self.exec()

    def _on_progress(self, percent: int, message: str):
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _on_finished(self, path: str):
        self._status_label.setText("Download complete!")
        self._cancel_btn.setText(t("common.close"))
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)

    def _on_error(self, message: str):
        self._status_label.setText(f"Error: {message}")
        self._status_label.setStyleSheet("font-size: 12px; color: #EF4444;")
        self._cancel_btn.setText(t("common.close"))
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        self.reject()
