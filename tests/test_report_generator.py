"""Tests for core.report_generator module."""

import json
from pathlib import Path

import pytest

from core.report_generator import ReportGenerator


class TestGenerateJson:
    def test_creates_file(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.json")
        result = gen.generate_json(sample_analysis_result, output)
        assert result is True
        assert Path(output).exists()

    def test_json_structure(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.json")
        gen.generate_json(sample_analysis_result, output)

        with open(output) as f:
            data = json.load(f)

        assert data["tool"] == "LocalMedScan"
        assert data["screening_type"] == "xray"
        assert data["model_name"] == "densenet121-all"
        assert len(data["findings"]) == 3
        assert "disclaimer" in data
        assert data["overall_confidence"] == 0.78

    def test_findings_content(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.json")
        gen.generate_json(sample_analysis_result, output)

        with open(output) as f:
            data = json.load(f)

        finding = data["findings"][0]
        assert finding["name"] == "Pneumonia"
        assert finding["confidence"] == 0.78
        assert finding["severity"] == "high"


class TestGenerateTxt:
    def test_creates_file(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.txt")
        result = gen.generate_txt(sample_analysis_result, output)
        assert result is True
        assert Path(output).exists()

    def test_contains_findings(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.txt")
        gen.generate_txt(sample_analysis_result, output)

        content = Path(output).read_text()
        assert "Pneumonia" in content
        assert "Cardiomegaly" in content
        assert "78.0%" in content
        assert "LOCALMEDSCAN" in content
        assert "Svetozar Technologies" in content

    def test_contains_disclaimer(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.txt")
        gen.generate_txt(sample_analysis_result, output)

        content = Path(output).read_text()
        assert "screening aid" in content.lower()

    def test_empty_findings(self, tmp_dir):
        from core.utils import AnalysisResult, ScreeningType
        result = AnalysisResult(success=True, screening_type=ScreeningType.MALARIA)
        gen = ReportGenerator()
        output = str(tmp_dir / "report.txt")
        success = gen.generate_txt(result, output)
        assert success
        content = Path(output).read_text()
        assert "No significant findings" in content


class TestGeneratePdf:
    def test_creates_file(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.pdf")
        result = gen.generate_pdf(sample_analysis_result, output)
        assert result is True
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_pdf_magic_bytes(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.pdf")
        gen.generate_pdf(sample_analysis_result, output)
        with open(output, "rb") as f:
            magic = f.read(4)
        assert magic == b"%PDF"

    def test_progress_callback(self, tmp_dir, sample_analysis_result):
        gen = ReportGenerator()
        output = str(tmp_dir / "report.pdf")
        steps = []
        gen.generate_pdf(
            sample_analysis_result, output,
            on_progress=lambda s, t, m: steps.append((s, t, m))
        )
        assert len(steps) == 4  # 4 progress steps
