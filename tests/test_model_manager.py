"""Tests for core.model_manager module."""

import pytest

from core.model_manager import ModelManager, get_model_manager, MODEL_REGISTRY
from core.utils import ScreeningType


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) >= 2

    def test_malaria_model_in_registry(self):
        names = [m.name for m in MODEL_REGISTRY]
        assert "malaria-mobilenetv2" in names

    def test_xray_model_in_registry(self):
        names = [m.name for m in MODEL_REGISTRY]
        assert "densenet121-all" in names

    def test_model_info_fields(self):
        for model in MODEL_REGISTRY:
            assert model.name
            assert model.display_name
            assert isinstance(model.screening_type, ScreeningType)
            assert model.size_mb > 0
            assert model.description


class TestModelManager:
    def test_get_registry(self):
        mm = ModelManager()
        registry = mm.get_registry()
        assert len(registry) == len(MODEL_REGISTRY)

    def test_get_model_info(self):
        mm = ModelManager()
        info = mm.get_model_info("malaria-mobilenetv2")
        assert info is not None
        assert info.screening_type == ScreeningType.MALARIA

    def test_get_model_info_unknown(self):
        mm = ModelManager()
        info = mm.get_model_info("nonexistent-model")
        assert info is None

    def test_get_models_for_type(self):
        mm = ModelManager()
        xray_models = mm.get_models_for_type(ScreeningType.XRAY)
        assert len(xray_models) >= 1
        assert all(m.screening_type == ScreeningType.XRAY for m in xray_models)

    def test_get_model_path(self):
        mm = ModelManager()
        path = mm.get_model_path("malaria-mobilenetv2")
        assert str(path).endswith("model.pth")
        assert "malaria-mobilenetv2" in str(path)

    def test_total_size_non_negative(self):
        mm = ModelManager()
        assert mm.get_total_size() >= 0

    def test_total_size_formatted(self):
        mm = ModelManager()
        formatted = mm.get_total_size_formatted()
        assert any(unit in formatted for unit in ("B", "KB", "MB", "GB"))

    def test_singleton(self):
        mm1 = get_model_manager()
        mm2 = get_model_manager()
        assert mm1 is mm2

    def test_xray_model_always_available(self):
        mm = ModelManager()
        # X-ray model auto-downloads via torchxrayvision library
        assert mm.is_model_available("densenet121-all") is True
