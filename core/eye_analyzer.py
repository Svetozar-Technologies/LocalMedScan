"""Diabetic retinopathy screening — Phase 2 stub."""

from core.utils import AnalysisConfig, AnalysisResult, CancelCheck, ProgressCallback
from typing import Optional


class EyeAnalyzer:
    """Placeholder for diabetic retinopathy severity grading (ResNet50 on EyePACS).

    Will be implemented in Phase 2 with:
    - 5-level severity grading (No DR, Mild, Moderate, Severe, Proliferative)
    - ResNet50 backbone
    - ~100 MB model size
    - 79-95% accuracy on EyePACS dataset
    """

    def analyze(
        self,
        config: AnalysisConfig,
        on_progress: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
    ) -> AnalysisResult:
        return AnalysisResult(
            success=False,
            screening_type=config.screening_type,
            error_message="Diabetic retinopathy screening is coming in a future update.",
            input_path=config.input_path,
        )
