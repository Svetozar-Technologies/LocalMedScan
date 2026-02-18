"""Side-by-side display of original image and Grad-CAM heatmap overlay."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from i18n import t


class HeatmapOverlay(QWidget):
    """Displays original image alongside the Grad-CAM attention heatmap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.hide()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        title = QLabel(t("results.heatmap"))
        title.setProperty("class", "sectionTitle")
        title.setStyleSheet("font-size: 16px;")

        desc = QLabel(t("results.heatmap_desc"))
        desc.setProperty("class", "sectionSubtitle")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px;")

        images_row = QHBoxLayout()
        images_row.setSpacing(16)

        # Original image
        original_col = QVBoxLayout()
        original_label = QLabel(t("results.original"))
        original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_label.setProperty("class", "imageLabel")
        self._original_image = QLabel()
        self._original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._original_image.setMinimumSize(200, 200)
        original_col.addWidget(original_label)
        original_col.addWidget(self._original_image)

        # Heatmap overlay
        heatmap_col = QVBoxLayout()
        heatmap_label = QLabel(t("results.heatmap_overlay"))
        heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heatmap_label.setProperty("class", "imageLabel")
        self._heatmap_image = QLabel()
        self._heatmap_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heatmap_image.setMinimumSize(200, 200)
        heatmap_col.addWidget(heatmap_label)
        heatmap_col.addWidget(self._heatmap_image)

        images_row.addLayout(original_col)
        images_row.addLayout(heatmap_col)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(images_row)

    def set_images(self, original_path: str, heatmap_path: str):
        """Load and display the original and heatmap images."""
        img_size = 224

        original_pixmap = QPixmap(original_path)
        if not original_pixmap.isNull():
            self._original_image.setPixmap(original_pixmap.scaled(
                img_size, img_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

        heatmap_pixmap = QPixmap(heatmap_path)
        if not heatmap_pixmap.isNull():
            self._heatmap_image.setPixmap(heatmap_pixmap.scaled(
                img_size, img_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

        self.show()

    def reset(self):
        """Clear images and hide."""
        self._original_image.clear()
        self._heatmap_image.clear()
        self.hide()
