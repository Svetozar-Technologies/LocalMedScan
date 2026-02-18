"""Model registry, download, caching, and verification."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.utils import ScreeningType, format_file_size, get_models_dir


@dataclass
class ModelInfo:
    """Metadata about an available AI model."""
    name: str
    display_name: str
    screening_type: ScreeningType
    size_mb: float
    description: str
    auto_download: bool = True
    accuracy: str = ""


MODEL_REGISTRY: List[ModelInfo] = [
    ModelInfo(
        name="malaria-mobilenetv2",
        display_name="MobileNetV2 (Malaria)",
        screening_type=ScreeningType.MALARIA,
        size_mb=14.0,
        description="Malaria parasite detection from blood smear images. Trained on NIH Malaria Dataset (27,558 images).",
        auto_download=True,
        accuracy="95-97%",
    ),
    ModelInfo(
        name="densenet121-all",
        display_name="DenseNet121 (Chest X-Ray)",
        screening_type=ScreeningType.XRAY,
        size_mb=50.0,
        description="TB, pneumonia, and 12 other conditions. Trained on NIH ChestX-ray14 + CheXpert + MIMIC-CXR.",
        auto_download=True,
        accuracy="85-95% (varies by condition)",
    ),
]


class ModelManager:
    """Manages AI model downloads, caching, and lifecycle."""

    def __init__(self):
        self._models_dir = get_models_dir()

    def get_registry(self) -> List[ModelInfo]:
        """Get all available models."""
        return MODEL_REGISTRY

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get info for a specific model."""
        for model in MODEL_REGISTRY:
            if model.name == model_name:
                return model
        return None

    def get_models_for_type(self, screening_type: ScreeningType) -> List[ModelInfo]:
        """Get models available for a screening type."""
        return [m for m in MODEL_REGISTRY if m.screening_type == screening_type]

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is downloaded and ready.

        For auto_download models (TorchXRayVision), they download on first use
        via the library's built-in caching, so we consider them always available.
        For custom models (malaria), check the local models directory.
        """
        info = self.get_model_info(model_name)
        if info and info.auto_download:
            # TorchXRayVision models auto-download via the library
            if info.screening_type == ScreeningType.XRAY:
                return True
            # Malaria model: check if weights file exists
            if info.screening_type == ScreeningType.MALARIA:
                model_path = self.get_model_path(model_name)
                return model_path.exists()
        return False

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model's weights file."""
        return self._models_dir / model_name / "model.pth"

    def get_models_dir(self) -> Path:
        """Get the root models directory."""
        return self._models_dir

    def get_total_size(self) -> int:
        """Get total size of all downloaded models in bytes."""
        total = 0
        if self._models_dir.exists():
            for f in self._models_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def get_total_size_formatted(self) -> str:
        """Get total model storage as human-readable string."""
        return format_file_size(self.get_total_size())

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        model_dir = self._models_dir / model_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False

    def ensure_model_dir(self, model_name: str) -> Path:
        """Create and return the directory for a model."""
        model_dir = self._models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir


# Module-level singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
