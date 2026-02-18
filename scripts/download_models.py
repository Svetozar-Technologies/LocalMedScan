#!/usr/bin/env python3
"""CLI utility to pre-download AI models for LocalMedScan.

Usage:
    python scripts/download_models.py          # Download all models
    python scripts/download_models.py --list   # List available models
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_manager import get_model_manager


def main():
    parser = argparse.ArgumentParser(description="Download LocalMedScan AI models.")
    parser.add_argument("--list", action="store_true", help="List available models.")
    args = parser.parse_args()

    mm = get_model_manager()

    if args.list:
        print("Available models:")
        print("-" * 60)
        for model in mm.get_registry():
            available = mm.is_model_available(model.name)
            status = "Downloaded" if available else "Not Downloaded"
            print(f"  {model.display_name}")
            print(f"    Name: {model.name}")
            print(f"    Type: {model.screening_type.value}")
            print(f"    Size: ~{model.size_mb:.0f} MB")
            print(f"    Accuracy: {model.accuracy}")
            print(f"    Status: {status}")
            print()
        total = mm.get_total_size_formatted()
        print(f"Total downloaded: {total}")
        return

    print("LocalMedScan Model Downloader")
    print("=" * 40)
    print()
    print("Note: TorchXRayVision models (densenet121-all) auto-download on first use.")
    print("The malaria model requires manual weight training or download.")
    print()

    for model in mm.get_registry():
        available = mm.is_model_available(model.name)
        if available:
            print(f"  [OK] {model.display_name} — already available")
        elif model.auto_download and model.screening_type.value == "xray":
            print(f"  [AUTO] {model.display_name} — will auto-download on first use")
        else:
            print(f"  [MISSING] {model.display_name} — requires manual setup")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
