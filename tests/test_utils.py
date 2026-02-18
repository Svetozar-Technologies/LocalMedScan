"""Tests for core.utils module."""

import os
import tempfile
from pathlib import Path

import pytest

from core.utils import (
    AnalysisConfig,
    AnalysisResult,
    Finding,
    ScreeningType,
    Severity,
    ValidationResult,
    format_file_size,
    get_data_dir,
    get_models_dir,
    get_cache_dir,
    get_platform,
    severity_from_confidence,
    validate_medical_image,
    SUPPORTED_IMAGE_EXTENSIONS,
)


class TestSeverityMapping:
    def test_high_confidence(self):
        assert severity_from_confidence(0.8) == Severity.HIGH
        assert severity_from_confidence(0.6) == Severity.HIGH
        assert severity_from_confidence(1.0) == Severity.HIGH

    def test_medium_confidence(self):
        assert severity_from_confidence(0.5) == Severity.MEDIUM
        assert severity_from_confidence(0.3) == Severity.MEDIUM

    def test_low_confidence(self):
        assert severity_from_confidence(0.1) == Severity.LOW
        assert severity_from_confidence(0.0) == Severity.LOW
        assert severity_from_confidence(0.29) == Severity.LOW


class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert format_file_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_zero(self):
        assert format_file_size(0) == "0 B"


class TestPlatformPaths:
    def test_data_dir_exists(self):
        d = get_data_dir()
        assert d.exists()
        assert d.is_dir()

    def test_models_dir_exists(self):
        d = get_models_dir()
        assert d.exists()
        assert d.is_dir()

    def test_cache_dir_exists(self):
        d = get_cache_dir()
        assert d.exists()
        assert d.is_dir()

    def test_platform(self):
        p = get_platform()
        assert p in ("macos", "windows", "linux")


class TestDataclasses:
    def test_finding(self):
        f = Finding(name="Test", confidence=0.5, severity=Severity.MEDIUM)
        assert f.name == "Test"
        assert f.description == ""

    def test_analysis_config_defaults(self):
        c = AnalysisConfig(input_path="/test.jpg", screening_type=ScreeningType.XRAY)
        assert c.model_name == "densenet121-all"
        assert c.generate_heatmap is True
        assert c.confidence_threshold == 0.3

    def test_analysis_result_defaults(self):
        r = AnalysisResult(success=True)
        assert r.findings == []
        assert r.overall_confidence == 0.0
        assert "screening aid" in r.disclaimer

    def test_screening_types(self):
        assert ScreeningType.MALARIA.value == "malaria"
        assert ScreeningType.XRAY.value == "xray"
        assert ScreeningType.SKIN.value == "skin"
        assert ScreeningType.EYE.value == "eye"

    def test_validation_result_defaults(self):
        v = ValidationResult(valid=True)
        assert v.error_message == ""
        assert v.is_dicom is False


class TestValidation:
    def test_empty_path(self):
        result = validate_medical_image("")
        assert not result.valid

    def test_nonexistent_file(self):
        result = validate_medical_image("/nonexistent/file.jpg")
        assert not result.valid

    def test_unsupported_format(self, tmp_dir):
        path = tmp_dir / "test.xyz"
        path.write_text("fake content")
        result = validate_medical_image(str(path))
        assert not result.valid

    def test_empty_file(self, tmp_dir):
        path = tmp_dir / "empty.jpg"
        path.touch()
        result = validate_medical_image(str(path))
        assert not result.valid

    def test_valid_png(self, sample_rgb_image):
        result = validate_medical_image(sample_rgb_image)
        assert result.valid
        assert result.image_width == 224
        assert result.image_height == 224
        assert result.file_size_bytes > 0

    def test_valid_grayscale(self, sample_grayscale_image):
        result = validate_medical_image(sample_grayscale_image)
        assert result.valid

    def test_supported_extensions(self):
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".dcm" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".gif" not in SUPPORTED_IMAGE_EXTENSIONS
