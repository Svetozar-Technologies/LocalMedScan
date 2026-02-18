"""Medical image loading, preprocessing, and normalization."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from core.utils import CancelCheck, ProgressCallback


class ImagePreprocessor:
    """Handles loading and preprocessing of medical images for model input."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load an image file as a numpy array.

        Supports JPEG, PNG, BMP, TIFF. For DICOM, use load_dicom().
        """
        path = Path(image_path)
        if path.suffix.lower() == ".dcm":
            return ImagePreprocessor.load_dicom(image_path)

        img = Image.open(image_path)
        return np.array(img)

    @staticmethod
    def load_dicom(dicom_path: str) -> np.ndarray:
        """Load a DICOM file and extract pixel data as uint8 array."""
        import pydicom

        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply window/level if available
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
            width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
            lower = center - width / 2
            upper = center + width / 2
            pixel_array = np.clip(pixel_array, lower, upper)

        # Normalize to 0-255
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        return pixel_array

    @staticmethod
    def preprocess_for_malaria(image_path: str) -> "torch.Tensor":
        """Preprocess a blood smear image for the malaria MobileNetV2 model.

        Returns tensor of shape (1, 3, 224, 224) with ImageNet normalization.
        """
        import torch
        from torchvision import transforms

        img = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        tensor = transform(img)
        return tensor.unsqueeze(0)  # Add batch dimension

    @staticmethod
    def preprocess_for_xray(image_path: str) -> "torch.Tensor":
        """Preprocess a chest X-ray image for TorchXRayVision models.

        Returns tensor of shape (1, 1, 224, 224) with txrv normalization.
        """
        import torch
        import torchxrayvision as xrv

        # Load image
        path = Path(image_path)
        if path.suffix.lower() == ".dcm":
            img_array = ImagePreprocessor.load_dicom(image_path)
        else:
            img = Image.open(image_path).convert("L")  # Grayscale
            img_array = np.array(img)

        # Ensure 2D grayscale
        if img_array.ndim == 3:
            img_array = img_array[:, :, 0]

        # Normalize to [0, 255] float
        img_array = img_array.astype(np.float32)

        # Resize to 224x224 using xrv's normalize function
        from skimage.transform import resize
        img_resized = resize(img_array, (224, 224), preserve_range=True, anti_aliasing=True)

        # TorchXRayVision expects values in [-1024, 1024] range
        # Normalize: scale [0,255] to [-1024, 1024]
        img_normalized = (img_resized / 255.0) * 2048 - 1024

        # Shape: (1, 1, 224, 224)
        tensor = torch.from_numpy(img_normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor

    @staticmethod
    def create_thumbnail(image_path: str, size: Tuple[int, int] = (128, 128)) -> bytes:
        """Create a JPEG thumbnail and return as bytes."""
        import io

        path = Path(image_path)
        if path.suffix.lower() == ".dcm":
            img_array = ImagePreprocessor.load_dicom(image_path)
            img = Image.fromarray(img_array)
        else:
            img = Image.open(image_path)

        img.thumbnail(size, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
