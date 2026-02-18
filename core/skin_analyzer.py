"""Skin lesion classification — Phase 2 stub."""

from core.utils import AnalysisConfig, AnalysisResult, CancelCheck, ProgressCallback
from typing import Optional


class SkinAnalyzer:
    """Placeholder for skin lesion classification (EfficientNet on HAM10000).

    Will be implemented in Phase 2 with:
    - 7-class classification (melanoma, BCC, BKL, DF, NV, VASC, AKIEC)
    - EfficientNet-B0 backbone
    - ~30 MB model size
    - 85-92% accuracy on HAM10000 dataset
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
            error_message="Skin lesion screening is coming in a future update.",
            input_path=config.input_path,
        )
