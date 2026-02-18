"""Tests for core.history_db module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.history_db import HistoryDB
from core.utils import AnalysisResult, Finding, ScreeningType, Severity


@pytest.fixture
def temp_db():
    """Create a temporary HistoryDB that uses a temp file."""
    with tempfile.TemporaryDirectory() as d:
        db_path = str(Path(d) / "test_history.db")
        with patch("core.history_db.get_history_db_path", return_value=Path(db_path)):
            db = HistoryDB()
            yield db


@pytest.fixture
def sample_result():
    return AnalysisResult(
        success=True,
        screening_type=ScreeningType.XRAY,
        findings=[
            Finding("Pneumonia", 0.78, Severity.HIGH, "Possible pneumonia."),
            Finding("Cardiomegaly", 0.45, Severity.MEDIUM, "Enlarged heart."),
        ],
        overall_confidence=0.78,
        model_name="densenet121-all",
        input_path="/fake/xray.jpg",
    )


class TestHistoryDB:
    def test_save_and_count(self, temp_db, sample_result):
        entry_id = temp_db.save(sample_result, b"thumbnail_data")
        assert entry_id > 0
        assert temp_db.count() == 1

    def test_save_multiple(self, temp_db, sample_result):
        temp_db.save(sample_result)
        temp_db.save(sample_result)
        temp_db.save(sample_result)
        assert temp_db.count() == 3

    def test_get_all(self, temp_db, sample_result):
        temp_db.save(sample_result)
        temp_db.save(sample_result)
        entries = temp_db.get_all()
        assert len(entries) == 2
        # Newest first
        assert entries[0].id > entries[1].id

    def test_get_all_limit(self, temp_db, sample_result):
        for _ in range(5):
            temp_db.save(sample_result)
        entries = temp_db.get_all(limit=3)
        assert len(entries) == 3

    def test_get_by_id(self, temp_db, sample_result):
        entry_id = temp_db.save(sample_result, b"thumb")
        entry = temp_db.get_by_id(entry_id)
        assert entry is not None
        assert entry.id == entry_id
        assert entry.screening_type == ScreeningType.XRAY
        assert entry.overall_confidence == 0.78
        assert entry.model_name == "densenet121-all"
        assert entry.thumbnail == b"thumb"

    def test_get_by_id_not_found(self, temp_db):
        entry = temp_db.get_by_id(99999)
        assert entry is None

    def test_get_by_type(self, temp_db):
        xray_result = AnalysisResult(
            success=True, screening_type=ScreeningType.XRAY,
            overall_confidence=0.5, model_name="xray", input_path="/x.jpg"
        )
        malaria_result = AnalysisResult(
            success=True, screening_type=ScreeningType.MALARIA,
            overall_confidence=0.9, model_name="malaria", input_path="/m.jpg"
        )
        temp_db.save(xray_result)
        temp_db.save(malaria_result)
        temp_db.save(xray_result)

        xray_entries = temp_db.get_by_type(ScreeningType.XRAY)
        assert len(xray_entries) == 2

        malaria_entries = temp_db.get_by_type(ScreeningType.MALARIA)
        assert len(malaria_entries) == 1

    def test_delete(self, temp_db, sample_result):
        entry_id = temp_db.save(sample_result)
        assert temp_db.count() == 1
        deleted = temp_db.delete(entry_id)
        assert deleted is True
        assert temp_db.count() == 0

    def test_delete_nonexistent(self, temp_db):
        deleted = temp_db.delete(99999)
        assert deleted is False

    def test_clear_all(self, temp_db, sample_result):
        for _ in range(5):
            temp_db.save(sample_result)
        assert temp_db.count() == 5
        count = temp_db.clear_all()
        assert count == 5
        assert temp_db.count() == 0

    def test_clear_all_empty(self, temp_db):
        count = temp_db.clear_all()
        assert count == 0

    def test_findings_count_stored(self, temp_db, sample_result):
        entry_id = temp_db.save(sample_result)
        entry = temp_db.get_by_id(entry_id)
        assert entry.findings_count == 2

    def test_findings_json_stored(self, temp_db, sample_result):
        import json
        entry_id = temp_db.save(sample_result)
        entry = temp_db.get_by_id(entry_id)
        findings = json.loads(entry.findings_json)
        assert len(findings) == 2
        assert findings[0]["name"] == "Pneumonia"

    def test_timestamp_format(self, temp_db, sample_result):
        entry_id = temp_db.save(sample_result)
        entry = temp_db.get_by_id(entry_id)
        # ISO format: YYYY-MM-DDTHH:MM:SS
        assert "T" in entry.timestamp
        assert len(entry.timestamp) >= 19
