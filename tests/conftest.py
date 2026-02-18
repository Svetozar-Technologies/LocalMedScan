"""Shared test fixtures for LocalMedScan."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.utils import (
    AnalysisResult,
    Finding,
    ScreeningType,
    Severity,
)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_rgb_image(tmp_dir):
    """Create a sample 224x224 RGB image (simulates a blood smear)."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = tmp_dir / "sample_rgb.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_grayscale_image(tmp_dir):
    """Create a sample 224x224 grayscale image (simulates a chest X-ray)."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))
    path = tmp_dir / "sample_gray.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_tiny_image(tmp_dir):
    """Create a tiny 32x32 image for thumbnail tests."""
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    path = tmp_dir / "tiny.jpg"
    img.save(path, format="JPEG")
    return str(path)


@pytest.fixture
def sample_analysis_result():
    """Create a sample AnalysisResult for testing."""
    return AnalysisResult(
        success=True,
        screening_type=ScreeningType.XRAY,
        findings=[
            Finding(
                name="Pneumonia",
                confidence=0.78,
                severity=Severity.HIGH,
                description="Possible pneumonia detected.",
            ),
            Finding(
                name="Cardiomegaly",
                confidence=0.45,
                severity=Severity.MEDIUM,
                description="Possible enlarged heart.",
            ),
            Finding(
                name="Effusion",
                confidence=0.22,
                severity=Severity.LOW,
                description="Minor pleural effusion.",
            ),
        ],
        overall_confidence=0.78,
        processing_time_ms=342,
        model_name="densenet121-all",
        input_path="/fake/xray.jpg",
    )


@pytest.fixture
def sample_malaria_result():
    """Create a sample malaria AnalysisResult."""
    return AnalysisResult(
        success=True,
        screening_type=ScreeningType.MALARIA,
        findings=[
            Finding(
                name="Parasitized",
                confidence=0.96,
                severity=Severity.HIGH,
                description="Malaria parasites detected.",
            ),
        ],
        overall_confidence=0.96,
        processing_time_ms=120,
        model_name="malaria-mobilenetv2",
        input_path="/fake/blood_smear.png",
    )


@pytest.fixture(autouse=True)
def _init_i18n():
    """Initialize i18n for all tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        import i18n
        i18n.init()
    except Exception:
        pass
