"""Background worker for downloading AI models."""

import urllib.request
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from core.model_manager import ModelInfo, get_model_manager


class ModelDownloadWorker(QThread):
    """Downloads a model file with progress reporting."""

    progress = pyqtSignal(int, int, str)  # downloaded_mb, total_mb, message
    finished = pyqtSignal(str)             # model_name
    error = pyqtSignal(str)                # error message

    def __init__(self, model_info: ModelInfo, url: str, parent=None):
        super().__init__(parent)
        self._model_info = model_info
        self._url = url
        self._cancelled = False

    def run(self):
        try:
            manager = get_model_manager()
            model_dir = manager.ensure_model_dir(self._model_info.name)
            output_path = model_dir / "model.pth"

            self.progress.emit(0, int(self._model_info.size_mb), "Starting download...")

            # Download with progress
            def reporthook(block_num, block_size, total_size):
                if self._cancelled:
                    raise InterruptedError("Download cancelled")
                downloaded = block_num * block_size
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024) if total_size > 0 else self._model_info.size_mb
                self.progress.emit(
                    int(downloaded_mb),
                    int(total_mb),
                    f"Downloading {self._model_info.display_name}...",
                )

            urllib.request.urlretrieve(self._url, str(output_path), reporthook)

            if not self._cancelled:
                self.finished.emit(self._model_info.name)

        except InterruptedError:
            # Clean up partial download
            output_path = Path(manager.get_model_path(self._model_info.name))
            if output_path.exists():
                output_path.unlink()
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"Download failed: {str(e)}")

    def cancel(self):
        self._cancelled = True
