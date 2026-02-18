"""SQLite database for screening history."""

import json
import sqlite3
from datetime import datetime
from typing import List, Optional

from core.utils import (
    AnalysisResult,
    Finding,
    HistoryEntry,
    ScreeningType,
    Severity,
    get_history_db_path,
)


class HistoryDB:
    """Manages persistent storage of screening results."""

    def __init__(self):
        self._db_path = str(get_history_db_path())
        self._init_db()

    def _init_db(self):
        """Create the history table if it doesn't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screening_type TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    overall_confidence REAL NOT NULL,
                    findings_count INTEGER NOT NULL,
                    findings_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    thumbnail BLOB
                )
            """)

    def save(self, result: AnalysisResult, thumbnail: bytes = b"") -> int:
        """Save a screening result and return its ID."""
        findings_data = [
            {
                "name": f.name,
                "confidence": f.confidence,
                "severity": f.severity.value,
                "description": f.description,
            }
            for f in result.findings
        ]

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO history
                   (screening_type, input_path, overall_confidence,
                    findings_count, findings_json, model_name, timestamp, thumbnail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.screening_type.value,
                    result.input_path,
                    result.overall_confidence,
                    len(result.findings),
                    json.dumps(findings_data),
                    result.model_name,
                    datetime.now().isoformat(),
                    thumbnail,
                ),
            )
            return cursor.lastrowid

    def get_all(self, limit: int = 100) -> List[HistoryEntry]:
        """Get all history entries, newest first."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_by_type(self, screening_type: ScreeningType) -> List[HistoryEntry]:
        """Get history entries filtered by screening type."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM history WHERE screening_type = ? ORDER BY id DESC",
                (screening_type.value,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_by_id(self, entry_id: int) -> Optional[HistoryEntry]:
        """Get a single history entry by ID."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM history WHERE id = ?", (entry_id,)
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def delete(self, entry_id: int) -> bool:
        """Delete a history entry."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Delete all history entries. Returns count deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("DELETE FROM history")
            return cursor.rowcount

    def count(self) -> int:
        """Get total number of history entries."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM history").fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_entry(row) -> HistoryEntry:
        """Convert a database row to a HistoryEntry."""
        return HistoryEntry(
            id=row[0],
            screening_type=ScreeningType(row[1]),
            input_path=row[2],
            overall_confidence=row[3],
            findings_count=row[4],
            model_name=row[6],
            timestamp=row[7],
            thumbnail=row[8] or b"",
            findings_json=row[5],
        )


# Module-level singleton
_history_db: Optional[HistoryDB] = None


def get_history_db() -> HistoryDB:
    """Get the global HistoryDB instance."""
    global _history_db
    if _history_db is None:
        _history_db = HistoryDB()
    return _history_db
