"""Background worker for report generation."""

from PyQt6.QtCore import QThread, pyqtSignal

from core.report_generator import ReportGenerator
from core.utils import AnalysisResult


class ReportWorker(QThread):
    """Generates a report in a background thread."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)            # output_path
    error = pyqtSignal(str)

    def __init__(self, result: AnalysisResult, output_path: str, format: str = "pdf", parent=None):
        super().__init__(parent)
        self._result = result
        self._output_path = output_path
        self._format = format

    def run(self):
        try:
            generator = ReportGenerator()

            if self._format == "pdf":
                success = generator.generate_pdf(
                    self._result, self._output_path,
                    on_progress=lambda s, t, m: self.progress.emit(s, t, m),
                )
            elif self._format == "json":
                success = generator.generate_json(self._result, self._output_path)
            elif self._format == "txt":
                success = generator.generate_txt(self._result, self._output_path)
            else:
                self.error.emit(f"Unknown format: {self._format}")
                return

            if success:
                self.finished.emit(self._output_path)
            else:
                self.error.emit(f"Failed to generate {self._format.upper()} report.")
        except Exception as e:
            self.error.emit(str(e))
