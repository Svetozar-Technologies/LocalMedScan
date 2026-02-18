"""Chest X-ray screening for TB, pneumonia, and 12 other conditions.

Uses TorchXRayVision DenseNet121 pretrained on NIH ChestX-ray14 + CheXpert + MIMIC-CXR.
14-pathology detection with Grad-CAM visual explanations.
Model auto-downloads on first use via the library's built-in caching (~50 MB).
"""

import time
from typing import Optional

import numpy as np

from core.image_preprocessor import ImagePreprocessor
from core.utils import (
    AnalysisConfig,
    AnalysisResult,
    CancelCheck,
    Finding,
    ProgressCallback,
    ScreeningType,
    severity_from_confidence,
)


class XRayAnalyzer:
    """Analyzes chest X-ray images for 14 pathological conditions."""

    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Fibrosis", "Hernia",
        "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
        "Pneumonia", "Pneumothorax",
    ]

    # Conditions of highest clinical importance for TB/pneumonia screening
    HIGH_PRIORITY = {"Pneumonia", "Infiltration", "Consolidation", "Effusion", "Mass", "Nodule"}

    _model = None  # Class-level cache

    def analyze(
        self,
        config: AnalysisConfig,
        on_progress: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
    ) -> AnalysisResult:
        """Run chest X-ray screening."""
        start_time = time.time()

        def report(step, total, msg):
            if on_progress:
                on_progress(step, total, msg)

        def cancelled():
            return is_cancelled and is_cancelled()

        try:
            # Step 1: Preprocess
            report(1, 5, "Preprocessing chest X-ray...")
            if cancelled():
                return self._cancelled_result(config)

            tensor = ImagePreprocessor.preprocess_for_xray(config.input_path)

            # Step 2: Load model
            report(2, 5, "Loading chest X-ray model...")
            if cancelled():
                return self._cancelled_result(config)

            model = self._load_model()

            # Step 3: Inference
            report(3, 5, "Screening for 14 conditions...")
            if cancelled():
                return self._cancelled_result(config)

            import torch
            with torch.inference_mode():
                outputs = model(tensor)
                # TorchXRayVision outputs raw logits; apply sigmoid for probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

            # Step 4: Generate heatmap
            heatmap_path = ""
            if config.generate_heatmap:
                report(4, 5, "Generating attention heatmap...")
                if not cancelled():
                    heatmap_path = self._generate_heatmap(
                        model, tensor, config.input_path
                    )

            # Step 5: Build findings
            report(5, 5, "Building results...")

            findings = []
            pathology_names = model.pathologies if hasattr(model, 'pathologies') else self.PATHOLOGIES

            for i, name in enumerate(pathology_names):
                if i < len(probabilities):
                    confidence = float(probabilities[i])
                    if confidence >= config.confidence_threshold:
                        # Clean up underscore names
                        display_name = name.replace("_", " ")
                        findings.append(Finding(
                            name=display_name,
                            confidence=confidence,
                            severity=severity_from_confidence(confidence),
                            description=self._get_condition_description(name),
                        ))

            # Sort by confidence descending
            findings.sort(key=lambda f: f.confidence, reverse=True)

            overall = max((f.confidence for f in findings), default=0.0)
            elapsed_ms = int((time.time() - start_time) * 1000)

            return AnalysisResult(
                success=True,
                screening_type=ScreeningType.XRAY,
                findings=findings,
                overall_confidence=overall,
                heatmap_path=heatmap_path,
                processing_time_ms=elapsed_ms,
                model_name="densenet121-all",
                input_path=config.input_path,
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                screening_type=ScreeningType.XRAY,
                error_message=str(e),
                input_path=config.input_path,
            )

    def _load_model(self):
        """Load or return cached TorchXRayVision DenseNet121."""
        if XRayAnalyzer._model is not None:
            return XRayAnalyzer._model

        import torchxrayvision as xrv

        # Load the "all" model trained on multiple datasets
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()

        XRayAnalyzer._model = model
        return model

    def _generate_heatmap(self, model, input_tensor, original_image_path: str) -> str:
        """Generate Grad-CAM heatmap for the highest-confidence pathology."""
        try:
            import torch
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from PIL import Image

            # DenseNet121 final conv layer
            target_layers = [model.features[-1]]

            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]

            # Load and prepare original image
            from pathlib import Path as _Path
            if _Path(original_image_path).suffix.lower() == ".dcm":
                img_array = ImagePreprocessor.load_dicom(original_image_path)
                img_rgb = np.stack([img_array] * 3, axis=-1)
            else:
                img = Image.open(original_image_path).convert("RGB")
                img_rgb = np.array(img)

            # Resize to match CAM
            from PIL import Image as _Image
            img_resized = _Image.fromarray(img_rgb).resize((224, 224))
            img_float = np.array(img_resized).astype(np.float32) / 255.0

            visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            # Save
            from core.utils import get_cache_dir
            heatmap_dir = get_cache_dir() / "heatmaps"
            heatmap_dir.mkdir(parents=True, exist_ok=True)

            heatmap_filename = f"xray_heatmap_{int(time.time())}.png"
            heatmap_path = str(heatmap_dir / heatmap_filename)

            _Image.fromarray(visualization).save(heatmap_path)
            return heatmap_path

        except Exception:
            return ""

    @staticmethod
    def _get_condition_description(condition_name: str) -> str:
        """Get a brief clinical description for a detected condition."""
        descriptions = {
            "Atelectasis": "Partial or complete collapse of the lung or a section of the lung.",
            "Cardiomegaly": "Enlarged heart, which may indicate heart disease or other conditions.",
            "Consolidation": "Region of lung tissue filled with fluid instead of air, often seen in pneumonia.",
            "Edema": "Excess fluid in the lungs, which can indicate heart failure.",
            "Effusion": "Abnormal fluid collection in the pleural space around the lungs.",
            "Emphysema": "Damage to the air sacs (alveoli) in the lungs, common in COPD.",
            "Fibrosis": "Scarring of lung tissue, which can restrict breathing.",
            "Hernia": "Protrusion of an organ through the chest wall or diaphragm.",
            "Infiltration": "Substance denser than air (fluid, cells) within the lung tissue. May indicate infection or TB.",
            "Mass": "An abnormal growth or lesion in the lung that requires further investigation.",
            "Nodule": "A small round growth in the lung. Most are benign but some require follow-up.",
            "Pleural_Thickening": "Thickening of the pleural membrane surrounding the lungs.",
            "Pneumonia": "Infection causing inflammation of the air sacs in the lungs.",
            "Pneumothorax": "Collapsed lung caused by air leaking into the space between lung and chest wall.",
        }
        return descriptions.get(condition_name, "")

    @staticmethod
    def _cancelled_result(config: AnalysisConfig) -> AnalysisResult:
        return AnalysisResult(
            success=False,
            screening_type=ScreeningType.XRAY,
            error_message="Analysis cancelled.",
            input_path=config.input_path,
        )
