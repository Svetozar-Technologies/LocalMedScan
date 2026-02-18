"""Table widget displaying screening findings with severity colors."""

from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.utils import Finding, Severity
from i18n import t


class FindingsTable(QWidget):
    """Displays findings in a sortable table with severity color coding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(t("results.findings"))
        title.setProperty("class", "sectionTitle")
        title.setStyleSheet("font-size: 16px;")

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels([
            t("results.condition"),
            t("results.confidence"),
            t("results.severity"),
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 120)
        self._table.setColumnWidth(2, 100)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)

        layout.addWidget(title)
        layout.addWidget(self._table)

    def set_findings(self, findings: List[Finding]):
        """Populate the table with findings."""
        self._table.setRowCount(len(findings))

        for row, finding in enumerate(findings):
            # Condition name
            name_item = QTableWidgetItem(finding.name)
            name_item.setToolTip(finding.description)
            self._table.setItem(row, 0, name_item)

            # Confidence as percentage
            confidence_pct = f"{finding.confidence * 100:.1f}%"
            conf_item = QTableWidgetItem(confidence_pct)
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 1, conf_item)

            # Severity badge
            severity_label = self._create_severity_label(finding.severity)
            self._table.setCellWidget(row, 2, severity_label)

        self._table.setMinimumHeight(min(40 + len(findings) * 40, 300))
        self.show()

    def _create_severity_label(self, severity: Severity) -> QLabel:
        """Create a color-coded severity badge."""
        text_map = {
            Severity.HIGH: t("results.severity_high"),
            Severity.MEDIUM: t("results.severity_medium"),
            Severity.LOW: t("results.severity_low"),
        }
        label = QLabel(text_map.get(severity, ""))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("severity", severity.value)

        # Force style refresh
        label.style().unpolish(label)
        label.style().polish(label)
        return label

    def reset(self):
        """Clear the table and hide."""
        self._table.setRowCount(0)
        self.hide()
