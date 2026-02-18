# LocalMedScan

**Free, offline AI medical image screening assistant.**

> AI Should Be Free. AI Should Be Private. AI Should Be Yours.

LocalMedScan is an open-source desktop application that screens medical images using AI models running entirely on your device. No internet required after the initial model download. No data leaves your computer. No account needed.

**IMPORTANT: This is a screening aid only, NOT a diagnostic tool. Always consult a qualified healthcare professional.**

## Screening Modules

| Module | Disease | Model | Accuracy | Status |
|--------|---------|-------|----------|--------|
| Malaria Detection | Malaria parasites | MobileNetV2 (NIH Dataset) | 95-97% | Available |
| TB / Pneumonia | 14 chest conditions | DenseNet121 (TorchXRayVision) | 85-95% | Available |
| Skin Lesion | Skin cancer classification | EfficientNet (HAM10000) | 85-92% | Coming Soon |
| Eye / Retina | Diabetic retinopathy | ResNet50 (EyePACS) | 79-95% | Coming Soon |

## Features

- 100% offline processing after one-time model download
- Malaria detection from blood smear microscopy images
- Chest X-ray screening for TB, pneumonia, and 12 other conditions
- Grad-CAM heatmap visualization showing model attention areas
- Export reports as PDF, JSON, or plain text
- Screening history with local SQLite storage
- 8 languages: English, Hindi, Russian, Chinese, Japanese, Spanish, French, Arabic
- Light and dark themes
- DICOM image support for X-rays

## Installation

### Requirements

- Python 3.10+
- ~500 MB disk space (including PyTorch CPU)

### Setup

```bash
git clone https://github.com/Svetozar-Technologies/LocalMedScan.git
cd LocalMedScan
pip install -r requirements.txt
python main.py
```

## How It Works

1. **First Launch**: Accept the medical disclaimer and download AI models (~50 MB one-time)
2. **Select Module**: Choose Malaria Detection or TB/Pneumonia from the sidebar
3. **Load Image**: Drag and drop or browse for a medical image
4. **Analyze**: Click the analyze button. Processing takes <500ms on most laptops
5. **Review Results**: See findings with confidence scores, severity levels, and heatmap overlay
6. **Export**: Save results as PDF/JSON/TXT reports

## Architecture

```
core/           Pure business logic (no UI imports, portable to Android via ONNX)
workers/        QThread background workers
ui/             PyQt6 widgets and dialogs
i18n/           8-language translation files
assets/styles/  Light and dark QSS themes
models/         Downloaded AI models (gitignored)
```

## Privacy

- All image processing happens locally on your CPU
- No telemetry, analytics, or data collection
- No internet required after model download
- History stored in local SQLite database
- No account or registration required

## Target Users

- Frontline health workers in rural areas
- Medical students and educators
- Rural clinics with limited specialist access
- Underserved populations in India, Russia, and developing countries

## Medical Disclaimer

This application is a screening aid only. It is NOT a medical device and is NOT intended to diagnose, treat, cure, or prevent any disease. Results should ALWAYS be reviewed by a qualified healthcare professional. Do not make medical decisions based solely on this tool's output. If you have a medical emergency, contact your local emergency services immediately.

## License

MIT License - See [LICENSE](LICENSE)

## Credits

- **Organization**: [Svetozar Technologies](https://github.com/Svetozar-Technologies)
- **Mission**: AI Should Be Free. AI Should Be Private. AI Should Be Yours.
- **Models**: NIH Malaria Dataset, TorchXRayVision (NIH ChestX-ray14 + CheXpert + MIMIC-CXR)
