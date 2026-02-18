"""Progress bar widget with cancel button for long-running operations."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i18n import t


class ProgressWidget(QWidget):
    """Progress indicator with percentage, status message, and cancel button."""

    cancel_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(12)

        self._bar = QProgressBar()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)

        self._pct_label = QLabel("0%")
        self._pct_label.setProperty("class", "progressPercent")
        self._pct_label.setFixedWidth(45)

        bar_row.addWidget(self._bar, 1)
        bar_row.addWidget(self._pct_label)

        status_row = QHBoxLayout()
        status_row.setSpacing(12)

        self._status_label = QLabel("")
        self._status_label.setProperty("class", "progressStatus")

        self._cancel_btn = QPushButton(t("progress.cancel"))
        self._cancel_btn.setProperty("class", "cancelButton")
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.clicked.connect(self.cancel_clicked.emit)

        status_row.addWidget(self._status_label, 1)
        status_row.addWidget(self._cancel_btn)

        layout.addLayout(bar_row)
        layout.addLayout(status_row)

    def start(self):
        """Show and reset the progress widget."""
        self._bar.setValue(0)
        self._pct_label.setText("0%")
        self._status_label.setText("")
        self._cancel_btn.setEnabled(True)
        self.show()

    def update_progress(self, current: int, total: int, message: str):
        """Update progress bar, percentage, and status message."""
        pct = int(current / total * 100) if total > 0 else 0
        pct = min(pct, 100)
        self._bar.setValue(pct)
        self._pct_label.setText(f"{pct}%")
        self._status_label.setText(message)

    def finish(self):
        """Mark progress as complete."""
        self._bar.setValue(100)
        self._pct_label.setText("100%")
        self._cancel_btn.setEnabled(False)
        self._status_label.setText(t("progress.complete"))

    def reset(self):
        """Hide and reset the widget."""
        self._bar.setValue(0)
        self._pct_label.setText("0%")
        self._status_label.setText("")
        self.hide()
