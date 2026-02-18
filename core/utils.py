"""Shared utilities, dataclasses, validation, and platform-specific paths."""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List


# --- Type aliases ---

ProgressCallback = Callable[[int, int, str], None]  # (step, total, message)
CancelCheck = Callable[[], bool]  # Returns True if cancelled


# --- Enums ---

class ScreeningType(Enum):
    MALARIA = "malaria"
    XRAY = "xray"
    SKIN = "skin"
    EYE = "eye"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Dataclasses ---

@dataclass
class Finding:
    """A single finding from a medical image screening."""
    name: str
    confidence: float
    severity: Severity
    description: str = ""


@dataclass
class AnalysisConfig:
    """Configuration for a screening analysis."""
    input_path: str
    screening_type: ScreeningType
    model_name: str = "densenet121-all"
    generate_heatmap: bool = True
    confidence_threshold: float = 0.3


@dataclass
class AnalysisResult:
    """Result of a screening analysis."""
    success: bool
    screening_type: ScreeningType = ScreeningType.XRAY
    findings: list = field(default_factory=list)
    overall_confidence: float = 0.0
    heatmap_path: str = ""
    processing_time_ms: int = 0
    model_name: str = ""
    input_path: str = ""
    error_message: str = ""
    disclaimer: str = (
        "This is a screening aid, NOT a diagnostic tool. "
        "Always consult a qualified healthcare professional."
    )


@dataclass
class ValidationResult:
    """Result of image validation."""
    valid: bool
    error_message: str = ""
    image_width: int = 0
    image_height: int = 0
    file_size_bytes: int = 0
    is_dicom: bool = False


@dataclass
class HistoryEntry:
    """A saved screening history entry."""
    id: int
    screening_type: ScreeningType
    input_path: str
    overall_confidence: float
    findings_count: int
    model_name: str
    timestamp: str
    thumbnail: bytes = b""
    findings_json: str = ""


# --- Severity mapping ---

def severity_from_confidence(confidence: float) -> Severity:
    """Map confidence value to severity level."""
    if confidence >= 0.6:
        return Severity.HIGH
    elif confidence >= 0.3:
        return Severity.MEDIUM
    return Severity.LOW


# --- Platform-specific paths ---

def get_data_dir() -> Path:
    """Get the platform-specific application data directory."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "LocalMedScan"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home())) / "LocalMedScan"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "localmedscan"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_models_dir() -> Path:
    """Get the directory for downloaded AI models."""
    models_dir = get_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_cache_dir() -> Path:
    """Get the platform-specific cache directory."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "LocalMedScan"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LocalMedScan" / "Cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "localmedscan"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_history_db_path() -> Path:
    """Get the path to the SQLite history database."""
    return get_data_dir() / "history.db"


# --- Asset paths ---

def get_asset_path(relative_path: str) -> str:
    """Get absolute path to an asset file, handling PyInstaller frozen apps."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent.parent
    return str(base / relative_path)


# --- Validation ---

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}


def validate_medical_image(file_path: str) -> ValidationResult:
    """Validate that a file is a supported medical image."""
    from i18n import t

    if not file_path:
        return ValidationResult(valid=False, error_message=t("validation.no_file"))

    path = Path(file_path)

    if not path.exists():
        return ValidationResult(valid=False, error_message=t("validation.file_not_found"))

    if not path.is_file():
        return ValidationResult(valid=False, error_message=t("validation.not_a_file"))

    file_size = path.stat().st_size
    if file_size == 0:
        return ValidationResult(valid=False, error_message=t("validation.empty_file"))

    ext = path.suffix.lower()
    is_dicom = ext == ".dcm"

    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return ValidationResult(
            valid=False,
            error_message=t("validation.unsupported_format", ext=ext),
        )

    # Try to read image dimensions
    width, height = 0, 0
    try:
        if is_dicom:
            import pydicom
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            width = int(ds.Columns)
            height = int(ds.Rows)
        else:
            from PIL import Image
            with Image.open(str(path)) as img:
                width, height = img.size
    except Exception:
        return ValidationResult(
            valid=False,
            error_message=t("validation.cannot_read_image"),
        )

    return ValidationResult(
        valid=True,
        image_width=width,
        image_height=height,
        file_size_bytes=file_size,
        is_dicom=is_dicom,
    )


# --- Formatting ---

def format_file_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_platform() -> str:
    """Get the current platform name."""
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform == "win32":
        return "windows"
    return "linux"
