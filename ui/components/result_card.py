"""Medical screening result card with findings, heatmap, and export buttons."""

from typing import List

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.utils import AnalysisResult, Finding
from i18n import t
from ui.components.confidence_gauge import ConfidenceGauge
from ui.components.disclaimer_banner import DisclaimerBanner
from ui.components.findings_table import FindingsTable
from ui.components.heatmap_overlay import HeatmapOverlay


class ResultCard(QWidget):
    """Displays complete screening results with export options."""

    export_pdf = pyqtSignal(object)    # AnalysisResult
    export_json = pyqtSignal(object)   # AnalysisResult
    export_txt = pyqtSignal(object)    # AnalysisResult
    analyze_another = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self._result = None
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header row: gauge + title + metadata
        header_row = QHBoxLayout()
        header_row.setSpacing(20)

        self._gauge = ConfidenceGauge(label=t("results.overall_confidence"), size=120)

        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        self._title_label = QLabel(t("results.title"))
        self._title_label.setProperty("class", "sectionTitle")
        self._title_label.setStyleSheet("font-size: 18px;")

        self._findings_count_label = QLabel("")
        self._findings_count_label.setProperty("class", "sectionSubtitle")

        self._meta_label = QLabel("")
        self._meta_label.setProperty("class", "sectionSubtitle")
        self._meta_label.setStyleSheet("font-size: 12px; color: #888;")

        info_col.addWidget(self._title_label)
        info_col.addWidget(self._findings_count_label)
        info_col.addWidget(self._meta_label)
        info_col.addStretch()

        header_row.addWidget(self._gauge)
        header_row.addLayout(info_col, 1)

        # Findings table
        self._findings_table = FindingsTable()

        # Heatmap overlay
        self._heatmap_overlay = HeatmapOverlay()

        # Disclaimer (always visible in results)
        self._disclaimer = DisclaimerBanner()

        # Export buttons row
        export_row = QHBoxLayout()
        export_row.setSpacing(8)

        self._pdf_btn = QPushButton(t("results.export_pdf"))
        self._pdf_btn.setProperty("class", "secondaryButton")
        self._pdf_btn.clicked.connect(lambda: self.export_pdf.emit(self._result))

        self._json_btn = QPushButton(t("results.export_json"))
        self._json_btn.setProperty("class", "secondaryButton")
        self._json_btn.clicked.connect(lambda: self.export_json.emit(self._result))

        self._txt_btn = QPushButton(t("results.export_txt"))
        self._txt_btn.setProperty("class", "secondaryButton")
        self._txt_btn.clicked.connect(lambda: self.export_txt.emit(self._result))

        export_row.addWidget(self._pdf_btn)
        export_row.addWidget(self._json_btn)
        export_row.addWidget(self._txt_btn)
        export_row.addStretch()

        self._another_btn = QPushButton(t("results.analyze_another"))
        self._another_btn.setObjectName("primaryButton")
        self._another_btn.clicked.connect(self.analyze_another.emit)
        export_row.addWidget(self._another_btn)

        layout.addLayout(header_row)
        layout.addWidget(self._findings_table)
        layout.addWidget(self._heatmap_overlay)
        layout.addWidget(self._disclaimer)
        layout.addLayout(export_row)

    def show_result(self, result: AnalysisResult):
        """Display screening results."""
        self._result = result

        # Gauge
        self._gauge.set_score(result.overall_confidence)

        # Count
        count = len(result.findings)
        self._findings_count_label.setText(
            t("xray.findings_count", count=count) if count > 0
            else t("xray.no_findings")
        )

        # Metadata
        meta_parts = []
        if result.processing_time_ms:
            meta_parts.append(t("results.processing_time", time=result.processing_time_ms))
        if result.model_name:
            meta_parts.append(t("results.model_used", model=result.model_name))
        self._meta_label.setText("  |  ".join(meta_parts))

        # Findings
        if result.findings:
            self._findings_table.set_findings(result.findings)
        else:
            self._findings_table.reset()

        # Heatmap
        if result.heatmap_path:
            self._heatmap_overlay.set_images(result.input_path, result.heatmap_path)
        else:
            self._heatmap_overlay.reset()

        self.show()

    def reset(self):
        """Clear results and hide."""
        self._result = None
        self._gauge.reset()
        self._findings_table.reset()
        self._heatmap_overlay.reset()
        self.hide()
