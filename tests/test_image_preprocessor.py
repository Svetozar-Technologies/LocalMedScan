"""Tests for core.image_preprocessor module."""

import numpy as np
import pytest
from PIL import Image

from core.image_preprocessor import ImagePreprocessor


class TestLoadImage:
    def test_load_rgb(self, sample_rgb_image):
        arr = ImagePreprocessor.load_image(sample_rgb_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (224, 224, 3)

    def test_load_grayscale(self, sample_grayscale_image):
        arr = ImagePreprocessor.load_image(sample_grayscale_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (224, 224)


class TestPreprocessForMalaria:
    def test_output_shape(self, sample_rgb_image):
        tensor = ImagePreprocessor.preprocess_for_malaria(sample_rgb_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self, sample_rgb_image):
        import torch
        tensor = ImagePreprocessor.preprocess_for_malaria(sample_rgb_image)
        assert tensor.dtype == torch.float32

    def test_normalized_range(self, sample_rgb_image):
        tensor = ImagePreprocessor.preprocess_for_malaria(sample_rgb_image)
        # ImageNet normalization can produce negative values
        assert tensor.min() < 1.0
        assert tensor.max() < 10.0  # rough sanity check


class TestPreprocessForXray:
    def test_output_shape(self, sample_grayscale_image):
        tensor = ImagePreprocessor.preprocess_for_xray(sample_grayscale_image)
        assert tensor.shape == (1, 1, 224, 224)

    def test_output_range(self, sample_grayscale_image):
        tensor = ImagePreprocessor.preprocess_for_xray(sample_grayscale_image)
        # TorchXRayVision range: [-1024, 1024]
        assert tensor.min() >= -1025
        assert tensor.max() <= 1025


class TestCreateThumbnail:
    def test_returns_bytes(self, sample_rgb_image):
        result = ImagePreprocessor.create_thumbnail(sample_rgb_image)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_jpeg_format(self, sample_rgb_image):
        result = ImagePreprocessor.create_thumbnail(sample_rgb_image)
        # JPEG magic bytes
        assert result[:2] == b'\xff\xd8'

    def test_custom_size(self, sample_rgb_image):
        result = ImagePreprocessor.create_thumbnail(sample_rgb_image, size=(64, 64))
        assert isinstance(result, bytes)
        assert len(result) > 0
