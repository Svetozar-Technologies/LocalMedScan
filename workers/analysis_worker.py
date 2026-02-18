"""Generic analysis worker that dispatches to the correct analyzer."""

from PyQt6.QtCore import QThread, pyqtSignal

from core.utils import AnalysisConfig, AnalysisResult, ScreeningType


class AnalysisWorker(QThread):
    """Background worker for running medical image analysis."""

    progress = pyqtSignal(int, int, str)  # step, total, message
    finished = pyqtSignal(object)          # AnalysisResult
    error = pyqtSignal(str)                # error message

    def __init__(self, config: AnalysisConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._cancelled = False

    def run(self):
        try:
            analyzer = self._get_analyzer()
            result = analyzer.analyze(
                self._config,
                on_progress=self._on_progress,
                is_cancelled=self._is_cancelled,
            )
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"Analysis failed: {str(e)}")

    def cancel(self):
        """Request cancellation of the analysis."""
        self._cancelled = True

    def _on_progress(self, step: int, total: int, message: str):
        if not self._cancelled:
            self.progress.emit(step, total, message)

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def _get_analyzer(self):
        """Create the appropriate analyzer for the screening type."""
        if self._config.screening_type == ScreeningType.MALARIA:
            from core.malaria_analyzer import MalariaAnalyzer
            return MalariaAnalyzer()
        elif self._config.screening_type == ScreeningType.XRAY:
            from core.xray_analyzer import XRayAnalyzer
            return XRayAnalyzer()
        elif self._config.screening_type == ScreeningType.SKIN:
            from core.skin_analyzer import SkinAnalyzer
            return SkinAnalyzer()
        elif self._config.screening_type == ScreeningType.EYE:
            from core.eye_analyzer import EyeAnalyzer
            return EyeAnalyzer()
        else:
            raise ValueError(f"Unknown screening type: {self._config.screening_type}")
