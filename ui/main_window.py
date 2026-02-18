"""Main application window with sidebar navigation and stacked content."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from i18n import t
from ui.theme import ThemeManager
from ui.history_widget import HistoryWidget
from ui.malaria_widget import MalariaWidget
from ui.settings_widget import SettingsWidget
from ui.xray_widget import XRayWidget


class _ComingSoonWidget(QWidget):
    """Placeholder widget for Phase 2 features."""

    def __init__(self, title_key: str, desc_key: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        icon = QLabel("\U0001f6a7")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(t(title_key))
        title.setProperty("class", "comingSoonTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc = QLabel(t(desc_key))
        desc.setProperty("class", "comingSoonDesc")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setMaximumWidth(400)

        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(desc)

    def cleanup(self):
        pass


class MainWindow(QMainWindow):
    """Main application window with sidebar and stacked content area."""

    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        self._theme_manager = theme_manager
        self._nav_buttons = []
        self.setWindowTitle(t("app.title"))
        self.setMinimumSize(900, 620)
        self.resize(1060, 700)
        self._setup_ui()
        self._setup_menu_bar()
        self._switch_tab(0)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # Content area
        self._stack = QStackedWidget()
        self._stack.setObjectName("contentArea")

        self._malaria_widget = MalariaWidget()
        self._xray_widget = XRayWidget()
        self._skin_widget = _ComingSoonWidget("skin.coming_soon", "skin.coming_soon_desc")
        self._eye_widget = _ComingSoonWidget("eye.coming_soon", "eye.coming_soon_desc")
        self._history_widget = HistoryWidget()
        self._settings_widget = SettingsWidget(self._theme_manager)

        self._stack.addWidget(self._malaria_widget)    # 0
        self._stack.addWidget(self._xray_widget)       # 1
        self._stack.addWidget(self._skin_widget)       # 2
        self._stack.addWidget(self._eye_widget)        # 3
        self._stack.addWidget(self._history_widget)    # 4
        self._stack.addWidget(self._settings_widget)   # 5

        main_layout.addWidget(self._stack, 1)

    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        # Logo
        logo = QLabel(t("app.title"))
        logo.setObjectName("sidebarLogo")
        layout.addWidget(logo)

        subtitle = QLabel(t("app.subtitle"))
        subtitle.setObjectName("sidebarSubtitle")
        subtitle.setStyleSheet("font-size: 11px; color: #888; padding-bottom: 12px;")
        layout.addWidget(subtitle)

        # Screening section
        screening_label = QLabel(t("sidebar.screening"))
        screening_label.setProperty("class", "sidebarSection")
        layout.addWidget(screening_label)

        nav_items = [
            (t("sidebar.malaria"), "\U0001fa78"),   # 0 — drop of blood
            (t("sidebar.xray"), "\U0001fa7b"),       # 1 — X-ray
            (t("sidebar.skin"), "\U0001fa79"),        # 2 — adhesive bandage
            (t("sidebar.eye"), "\U0001f441"),          # 3 — eye
        ]

        for i, (label, icon) in enumerate(nav_items):
            btn = QPushButton(f"  {icon}  {label}")
            btn.setProperty("class", "navButton")
            btn.clicked.connect(lambda checked, idx=i: self._switch_tab(idx))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addSpacing(12)

        # Tools section
        tools_label = QLabel(t("sidebar.tools"))
        tools_label.setProperty("class", "sidebarSection")
        layout.addWidget(tools_label)

        history_btn = QPushButton(f"  \U0001f4cb  {t('sidebar.history')}")
        history_btn.setProperty("class", "navButton")
        history_btn.clicked.connect(lambda: self._switch_tab(4))
        layout.addWidget(history_btn)
        self._nav_buttons.append(history_btn)

        layout.addStretch()

        # Settings at bottom
        settings_btn = QPushButton(f"  \u2699\ufe0f  {t('sidebar.settings')}")
        settings_btn.setProperty("class", "navButton")
        settings_btn.clicked.connect(lambda: self._switch_tab(5))
        layout.addWidget(settings_btn)
        self._nav_buttons.append(settings_btn)

        return sidebar

    def _switch_tab(self, index: int):
        """Change active tab and update button styling."""
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            is_active = i == index
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Refresh history when switching to the history tab
        if index == 4:
            self._history_widget.refresh()

    def _setup_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu(t("menu.file"))
        quit_action = QAction(t("menu.quit"), self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menu_bar.addMenu(t("menu.view"))
        toggle_theme = QAction(t("menu.toggle_dark_mode"), self)
        toggle_theme.setShortcut("Ctrl+D")
        toggle_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme)

        # Navigate menu
        nav_menu = menu_bar.addMenu(t("menu.navigate"))
        nav_shortcuts = [
            (t("sidebar.malaria"), "Ctrl+1", 0),
            (t("sidebar.xray"), "Ctrl+2", 1),
            (t("sidebar.skin"), "Ctrl+3", 2),
            (t("sidebar.eye"), "Ctrl+4", 3),
            (t("sidebar.history"), "Ctrl+5", 4),
            (t("sidebar.settings"), "Ctrl+,", 5),
        ]
        for label, shortcut, index in nav_shortcuts:
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked, i=index: self._switch_tab(i))
            nav_menu.addAction(action)

    def _toggle_theme(self):
        self._theme_manager.toggle_theme()

    def closeEvent(self, event):
        """Clean up all workers before closing."""
        widgets = [
            self._malaria_widget,
            self._xray_widget,
            self._skin_widget,
            self._eye_widget,
            self._history_widget,
            self._settings_widget,
        ]
        for w in widgets:
            w.cleanup()
        QApplication.processEvents()
        event.accept()
