"""Malaria blood smear screening tab."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from core.history_db import get_history_db
from core.image_preprocessor import ImagePreprocessor
from core.utils import AnalysisConfig, AnalysisResult, ScreeningType, validate_medical_image
from i18n import t
from ui.components.disclaimer_banner import DisclaimerBanner
from ui.components.image_drop_zone import ImageDropZone
from ui.components.progress_widget import ProgressWidget
from ui.components.result_card import ResultCard
from workers.analysis_worker import AnalysisWorker
from workers.report_worker import ReportWorker


class MalariaWidget(QWidget):
    """Malaria parasite detection from blood smear images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: AnalysisWorker = None
        self._report_worker: ReportWorker = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # Disclaimer banner
        self._disclaimer = DisclaimerBanner()

        # Title
        title = QLabel(t("malaria.title"))
        title.setProperty("class", "sectionTitle")

        subtitle = QLabel(t("malaria.subtitle"))
        subtitle.setProperty("class", "sectionSubtitle")
        subtitle.setWordWrap(True)

        # Drop zone
        self._drop_zone = ImageDropZone(
            accepted_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
            placeholder_text=t("malaria.drop_text"),
        )

        # Options
        options_widget = QWidget()
        options_layout = QHBoxLayout(options_widget)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(16)

        self._heatmap_check = QCheckBox(t("xray.generate_heatmap"))
        self._heatmap_check.setChecked(True)

        options_layout.addWidget(self._heatmap_check)
        options_layout.addStretch()

        # Analyze button
        self._analyze_btn = QPushButton(t("malaria.analyze_button"))
        self._analyze_btn.setObjectName("primaryButton")
        self._analyze_btn.setEnabled(False)

        # Accuracy note
        accuracy_label = QLabel(t("malaria.accuracy_note"))
        accuracy_label.setProperty("class", "sectionSubtitle")
        accuracy_label.setStyleSheet("font-size: 12px; font-style: italic;")
        accuracy_label.setWordWrap(True)

        # Progress
        self._progress = ProgressWidget()

        # Results
        self._result_card = ResultCard()

        layout.addWidget(self._disclaimer)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._drop_zone)
        layout.addWidget(options_widget)
        layout.addWidget(self._analyze_btn)
        layout.addWidget(accuracy_label)
        layout.addWidget(self._progress)
        layout.addWidget(self._result_card)
        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self._drop_zone.file_selected.connect(self._on_file_selected)
        self._drop_zone.file_removed.connect(self._on_file_removed)
        self._analyze_btn.clicked.connect(self._on_analyze)
        self._progress.cancel_clicked.connect(self._on_cancel)
        self._result_card.analyze_another.connect(self._on_another)
        self._result_card.export_pdf.connect(self._export_report_pdf)
        self._result_card.export_json.connect(self._export_report_json)
        self._result_card.export_txt.connect(self._export_report_txt)

    def _on_file_selected(self, path: str):
        self._analyze_btn.setEnabled(True)
        self._result_card.reset()

    def _on_file_removed(self):
        self._analyze_btn.setEnabled(False)
        self._result_card.reset()

    def _on_analyze(self):
        file_path = self._drop_zone.get_file_path()
        if not file_path:
            return

        validation = validate_medical_image(file_path)
        if not validation.valid:
            QMessageBox.warning(self, t("common.error"), validation.error_message)
            return

        self._analyze_btn.setEnabled(False)
        self._result_card.reset()
        self._progress.start()

        config = AnalysisConfig(
            input_path=file_path,
            screening_type=ScreeningType.MALARIA,
            model_name="malaria-mobilenetv2",
            generate_heatmap=self._heatmap_check.isChecked(),
        )

        self._worker = AnalysisWorker(config, parent=self)
        self._worker.progress.connect(self._progress.update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: AnalysisResult):
        self._progress.finish()
        if result.success:
            self._result_card.show_result(result)
            self._save_to_history(result)
        else:
            QMessageBox.warning(self, t("common.error"), result.error_message)
        self._analyze_btn.setEnabled(True)

    def _on_error(self, message: str):
        self._progress.reset()
        self._analyze_btn.setEnabled(True)
        QMessageBox.critical(self, t("common.error"), message)

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self._progress.reset()
        self._analyze_btn.setEnabled(True)

    def _on_another(self):
        self._drop_zone.reset()
        self._result_card.reset()
        self._progress.reset()
        self._analyze_btn.setEnabled(False)

    def _save_to_history(self, result: AnalysisResult):
        """Save screening result to history database."""
        try:
            thumbnail = ImagePreprocessor.create_thumbnail(result.input_path)
            db = get_history_db()
            db.save(result, thumbnail)
        except Exception:
            pass  # History saving is best-effort

    def _export_report(self, result: AnalysisResult, fmt: str, title: str, ext: str):
        """Launch a file dialog and generate a report in the given format."""
        if not result:
            return
        path, _ = QFileDialog.getSaveFileName(self, title, f"malaria_report.{ext}", f"*.{ext}")
        if not path:
            return
        self._report_worker = ReportWorker(result, path, format=fmt, parent=self)
        self._report_worker.finished.connect(
            lambda p: QMessageBox.information(self, t("common.success"), t("export.success"))
        )
        self._report_worker.error.connect(
            lambda e: QMessageBox.warning(self, t("common.error"), e)
        )
        self._report_worker.start()

    def _export_report_pdf(self, result):
        self._export_report(result, "pdf", t("export.save_pdf_title"), "pdf")

    def _export_report_json(self, result):
        self._export_report(result, "json", t("export.save_json_title"), "json")

    def _export_report_txt(self, result):
        self._export_report(result, "txt", t("export.save_txt_title"), "txt")

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            if not self._worker.wait(5000):
                self._worker.terminate()
                self._worker.wait(2000)
        if self._report_worker and self._report_worker.isRunning():
            self._report_worker.wait(3000)
