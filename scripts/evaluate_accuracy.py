#!/usr/bin/env python3
"""Evaluate model accuracy on test datasets.

Usage:
    python scripts/evaluate_accuracy.py --malaria   # Test malaria model
    python scripts/evaluate_accuracy.py --xray      # Test chest X-ray model
    python scripts/evaluate_accuracy.py --all       # Test both
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_malaria(data_dir: str, max_samples: int = 200):
    """Evaluate malaria model on NIH cell images dataset.

    Dataset structure:
        cell_images/
            Parasitized/   (infected cells)
            Uninfected/    (healthy cells)
    """
    from core.image_preprocessor import ImagePreprocessor

    print("=" * 60)
    print("MALARIA MODEL EVALUATION")
    print("Model: MobileNetV2 (pretrained ImageNet)")
    print("Dataset: NIH Malaria Cell Images")
    print("=" * 60)

    parasitized_dir = Path(data_dir) / "Parasitized"
    uninfected_dir = Path(data_dir) / "Uninfected"

    if not parasitized_dir.exists() or not uninfected_dir.exists():
        print(f"ERROR: Expected Parasitized/ and Uninfected/ in {data_dir}")
        print(f"  Parasitized exists: {parasitized_dir.exists()}")
        print(f"  Uninfected exists: {uninfected_dir.exists()}")
        return

    parasitized_files = sorted([f for f in parasitized_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    uninfected_files = sorted([f for f in uninfected_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])

    print(f"\nTotal Parasitized images: {len(parasitized_files)}")
    print(f"Total Uninfected images: {len(uninfected_files)}")

    # Sample subset for speed
    np.random.seed(42)
    n_per_class = min(max_samples // 2, len(parasitized_files), len(uninfected_files))

    para_sample = list(np.random.choice(len(parasitized_files), n_per_class, replace=False))
    unin_sample = list(np.random.choice(len(uninfected_files), n_per_class, replace=False))

    test_images = [(parasitized_files[i], "Parasitized") for i in para_sample] + \
                  [(uninfected_files[i], "Uninfected") for i in unin_sample]

    np.random.shuffle(test_images)

    print(f"\nEvaluating {len(test_images)} images ({n_per_class} per class)...")
    print("-" * 60)

    # Load model
    import torch
    from torchvision import models, transforms

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # For a pretrained-only model (no fine-tuned weights), we test the feature
    # extraction capability. The malaria_analyzer uses a custom head, but
    # since we don't have fine-tuned weights, we test with a simple approach.
    model.eval()

    # Use the analyzer directly
    from core.utils import AnalysisConfig, ScreeningType
    from core.malaria_analyzer import MalariaAnalyzer

    analyzer = MalariaAnalyzer()

    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    errors = 0

    start_time = time.time()

    for i, (img_path, true_label) in enumerate(test_images):
        try:
            config = AnalysisConfig(
                input_path=str(img_path),
                screening_type=ScreeningType.MALARIA,
                model_name="malaria-mobilenetv2",
                generate_heatmap=False,
            )
            result = analyzer.analyze(config)

            if not result.success:
                errors += 1
                continue

            # Check prediction
            if result.findings:
                predicted = result.findings[0].name
            else:
                predicted = "Uninfected"

            is_correct = (predicted == true_label)
            if is_correct:
                correct += 1

            # Confusion matrix
            if true_label == "Parasitized" and predicted == "Parasitized":
                true_positives += 1
            elif true_label == "Uninfected" and predicted == "Parasitized":
                false_positives += 1
            elif true_label == "Uninfected" and predicted == "Uninfected":
                true_negatives += 1
            elif true_label == "Parasitized" and predicted == "Uninfected":
                false_negatives += 1

            total += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{i+1}/{len(test_images)}] Running accuracy: {acc:.1f}% ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {img_path.name}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("MALARIA RESULTS")
    print("=" * 60)
    accuracy = correct / total * 100 if total > 0 else 0

    sensitivity = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) * 100 if (true_negatives + false_positives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0

    print(f"  Total evaluated: {total}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms per image)")
    print()
    print(f"  Accuracy:    {accuracy:.1f}%")
    print(f"  Sensitivity: {sensitivity:.1f}% (true positive rate)")
    print(f"  Specificity: {specificity:.1f}% (true negative rate)")
    print(f"  Precision:   {precision:.1f}%")
    print()
    print("  Confusion Matrix:")
    print(f"                   Predicted Positive  Predicted Negative")
    print(f"  Actual Positive  {true_positives:>18}  {false_negatives:>18}")
    print(f"  Actual Negative  {false_positives:>18}  {true_negatives:>18}")
    print()

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "total": total,
        "errors": errors,
    }


def evaluate_xray(data_dir: str, max_samples: int = 100):
    """Evaluate X-ray model on TB chest X-ray datasets.

    Uses TorchXRayVision DenseNet121 to evaluate on Montgomery/Shenzhen TB datasets.
    We check if the model flags relevant pathologies for TB-positive images.
    """
    print("=" * 60)
    print("CHEST X-RAY MODEL EVALUATION")
    print("Model: DenseNet121 (TorchXRayVision)")
    print("Dataset: Montgomery County / Shenzhen TB")
    print("=" * 60)

    data_path = Path(data_dir)

    # Try to find images and labels
    # Montgomery: MontgomerySet/CXR_png/ with clinical readings
    # Shenzhen: ChinaSet_AllFiles/CXR_png/

    image_dirs = []
    label_files = []

    # Search for image directories
    png_dirs = list(data_path.rglob("CXR_png"))
    if not png_dirs:
        # Try looking for image files directly
        png_dirs = list(data_path.rglob("*.png"))
        if png_dirs:
            image_dirs = [png_dirs[0].parent]
        else:
            print(f"ERROR: No image directories found in {data_dir}")
            print("Looking for contents...")
            for item in sorted(data_path.rglob("*"))[:20]:
                print(f"  {item}")
            return
    else:
        image_dirs = png_dirs

    # Collect all images with labels
    test_images = []

    for img_dir in image_dirs:
        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() == '.png'])

        # Try to find clinical readings for labels
        clinical_dir = img_dir.parent / "ClinicalReadings"

        for img_path in images:
            label = "unknown"

            # Check clinical reading file
            if clinical_dir.exists():
                reading_file = clinical_dir / (img_path.stem + ".txt")
                if reading_file.exists():
                    content = reading_file.read_text().lower()
                    if "normal" in content:
                        label = "normal"
                    else:
                        label = "abnormal"  # TB or other finding

            # Shenzhen naming convention: CHNCXR_XXXX_Y.png where Y=0 normal, Y=1 TB
            name = img_path.stem
            if "CHNCXR_" in name:
                parts = name.split("_")
                if len(parts) >= 3:
                    if parts[-1] == "0":
                        label = "normal"
                    elif parts[-1] == "1":
                        label = "abnormal"

            test_images.append((img_path, label))

    # Filter to labeled images only
    labeled = [(p, l) for p, l in test_images if l != "unknown"]
    unlabeled = [(p, l) for p, l in test_images if l == "unknown"]

    print(f"\nTotal images found: {len(test_images)}")
    print(f"  Labeled: {len(labeled)}")
    print(f"  Unlabeled: {len(unlabeled)}")

    if labeled:
        normal_count = sum(1 for _, l in labeled if l == "normal")
        abnormal_count = sum(1 for _, l in labeled if l == "abnormal")
        print(f"  Normal: {normal_count}, Abnormal (TB): {abnormal_count}")

    # Use labeled if available, otherwise evaluate all
    eval_images = labeled if labeled else test_images

    # Sample if too many
    np.random.seed(42)
    if len(eval_images) > max_samples:
        indices = np.random.choice(len(eval_images), max_samples, replace=False)
        eval_images = [eval_images[i] for i in indices]

    print(f"\nEvaluating {len(eval_images)} images...")
    print("-" * 60)

    from core.utils import AnalysisConfig, ScreeningType
    from core.xray_analyzer import XRayAnalyzer

    analyzer = XRayAnalyzer()

    # TB-related pathologies
    tb_indicators = {"Infiltration", "Consolidation", "Effusion", "Fibrosis",
                     "Pneumonia", "Mass", "Nodule", "Atelectasis", "Pleural_Thickening"}

    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    errors = 0
    all_findings = defaultdict(int)

    start_time = time.time()

    for i, (img_path, true_label) in enumerate(eval_images):
        try:
            config = AnalysisConfig(
                input_path=str(img_path),
                screening_type=ScreeningType.XRAY,
                model_name="densenet121-all",
                generate_heatmap=False,
                confidence_threshold=0.3,
            )
            result = analyzer.analyze(config)

            if not result.success:
                errors += 1
                continue

            # Count findings
            for finding in result.findings:
                all_findings[finding.name] += 1

            # Prediction: if any significant finding above threshold, predict abnormal
            has_significant = any(
                f.confidence >= 0.3 and f.name in tb_indicators
                for f in result.findings
            )
            predicted = "abnormal" if has_significant else "normal"

            if true_label != "unknown":
                is_correct = (predicted == true_label)
                if is_correct:
                    correct += 1

                if true_label == "abnormal" and predicted == "abnormal":
                    true_positives += 1
                elif true_label == "normal" and predicted == "abnormal":
                    false_positives += 1
                elif true_label == "normal" and predicted == "normal":
                    true_negatives += 1
                elif true_label == "abnormal" and predicted == "normal":
                    false_negatives += 1

            total += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{i+1}/{len(eval_images)}] Running accuracy: {acc:.1f}% ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {img_path.name}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("CHEST X-RAY RESULTS")
    print("=" * 60)

    labeled_total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = correct / labeled_total * 100 if labeled_total > 0 else 0

    sensitivity = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) * 100 if (true_negatives + false_positives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0

    print(f"  Total evaluated: {total}")
    print(f"  Labeled evaluated: {labeled_total}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(total,1)*1000:.0f}ms per image)")
    print()

    if labeled_total > 0:
        print(f"  Accuracy:    {accuracy:.1f}%")
        print(f"  Sensitivity: {sensitivity:.1f}% (detects TB when present)")
        print(f"  Specificity: {specificity:.1f}% (correctly identifies normal)")
        print(f"  Precision:   {precision:.1f}%")
        print()
        print("  Confusion Matrix:")
        print(f"                   Predicted Abnormal  Predicted Normal")
        print(f"  Actual Abnormal  {true_positives:>18}  {false_negatives:>16}")
        print(f"  Actual Normal    {false_positives:>18}  {true_negatives:>16}")

    print()
    print("  Most common findings detected:")
    for name, count in sorted(all_findings.items(), key=lambda x: -x[1])[:10]:
        pct = count / total * 100
        print(f"    {name}: {count} ({pct:.0f}% of images)")
    print()

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "total": total,
        "labeled_total": labeled_total,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LocalMedScan model accuracy.")
    parser.add_argument("--malaria", action="store_true", help="Evaluate malaria model")
    parser.add_argument("--xray", action="store_true", help="Evaluate chest X-ray model")
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--malaria-dir", default="test_data/malaria/cell_images",
                       help="Path to malaria cell_images directory")
    parser.add_argument("--xray-dir", default="test_data/xray",
                       help="Path to chest X-ray dataset directory")
    parser.add_argument("--max-samples", type=int, default=200,
                       help="Max samples per evaluation (default: 200)")
    args = parser.parse_args()

    if not (args.malaria or args.xray or args.all):
        args.all = True

    results = {}

    if args.malaria or args.all:
        results["malaria"] = evaluate_malaria(args.malaria_dir, args.max_samples)

    if args.xray or args.all:
        results["xray"] = evaluate_xray(args.xray_dir, args.max_samples)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        if res:
            print(f"  {name.upper()}: {res['accuracy']:.1f}% accuracy ({res['total']} images, {res['errors']} errors)")
    print()


if __name__ == "__main__":
    main()
