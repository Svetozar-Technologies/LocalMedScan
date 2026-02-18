"""Circular confidence gauge with animated fill."""

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRectF, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class ConfidenceGauge(QWidget):
    """Animated circular gauge displaying a confidence percentage."""

    COLOR_HIGH = QColor("#EF4444")    # Red — high concern
    COLOR_MEDIUM = QColor("#F59E0B")  # Amber — moderate
    COLOR_LOW = QColor("#22C55E")     # Green — low concern
    COLOR_BG = QColor("#E5E7EB")      # Track color

    def __init__(self, label: str = "", size: int = 120, parent=None):
        super().__init__(parent)
        self._label = label
        self._size = size
        self._score = 0.0
        self._animated_score = 0.0
        self.setFixedSize(size, size)

        self._animation = QPropertyAnimation(self, b"animatedScore")
        self._animation.setDuration(800)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_score(self, score: float):
        """Set the score (0.0 - 1.0) and animate to it."""
        self._score = max(0.0, min(1.0, score))
        self._animation.setStartValue(self._animated_score)
        self._animation.setEndValue(self._score)
        self._animation.start()

    def _get_animated_score(self) -> float:
        return self._animated_score

    def _set_animated_score(self, value: float):
        self._animated_score = value
        self.update()

    animatedScore = pyqtProperty(float, _get_animated_score, _set_animated_score)

    def _get_color(self, score: float) -> QColor:
        if score >= 0.6:
            return self.COLOR_HIGH
        elif score >= 0.3:
            return self.COLOR_MEDIUM
        return self.COLOR_LOW

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen_width = 10
        margin = pen_width / 2 + 4
        rect = QRectF(margin, margin, self._size - 2 * margin, self._size - 2 * margin)

        # Background arc
        bg_pen = QPen(self.COLOR_BG, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        # Foreground arc
        color = self._get_color(self._animated_score)
        fg_pen = QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(fg_pen)
        span = int(-270 * self._animated_score * 16)
        painter.drawArc(rect, 225 * 16, span)

        # Center text
        painter.setPen(QPen(color))
        pct_text = f"{int(self._animated_score * 100)}%"
        font = QFont()
        font.setPixelSize(int(self._size * 0.22))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, pct_text)

        # Label below percentage
        if self._label:
            painter.setPen(QPen(QColor("#888888")))
            label_font = QFont()
            label_font.setPixelSize(int(self._size * 0.1))
            painter.setFont(label_font)
            label_rect = QRectF(rect.x(), rect.center().y() + 12, rect.width(), 20)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._label)

        painter.end()

    def reset(self):
        """Reset gauge to zero."""
        self._animation.stop()
        self._animated_score = 0.0
        self._score = 0.0
        self.update()
