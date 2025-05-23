import sys
import os
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    QSizePolicy
)
from PyQt6.QtGui import QPainter, QLinearGradient, QColor
from PyQt6.QtCore import Qt


def resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller bundled app."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # In development, use the current directory
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class HomePage(QWidget):
    """Home page widget for the ASL Translator application."""

    def __init__(self):
        super().__init__()
        self.main_app: Optional['SignLanguageApp'] = None
        self.gradient: Optional[QLinearGradient] = None
        self._init_window()
        self._init_ui()

    def _init_window(self) -> None:
        """Initialize window properties."""
        self.setWindowTitle("ASL Translator - Home")
        # Removed showMaximized() from here to avoid premature call

    def _init_ui(self) -> None:
        """Initialize UI components and layout."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(50)

        # Welcome label
        welcome_label = QLabel("Welcome to the ASL Translator")
        welcome_label.setObjectName("welcomeLabel")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Start button
        start_button = QPushButton("Start Translation")
        start_button.setObjectName("startButton")
        start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        start_button.clicked.connect(self._launch_main_app)

        # Dynamic spacer sizing
        spacer_height = self.height() // 4  # 25% of window height
        layout.addSpacerItem(
            QSpacerItem(20, spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        layout.addWidget(welcome_label)
        layout.addWidget(start_button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacerItem(
            QSpacerItem(20, spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # Load stylesheet
        try:
            with open(resource_path("HomePage_Style.qss"), "r") as file:
                self.setStyleSheet(file.read())
        except FileNotFoundError:
            print("Warning: HomePage_Style.qss not found. Using default styles.")

        self.setLayout(layout)

    def _launch_main_app(self) -> None:
        """Launch the main application and hide the home page."""
        try:
            from MainGui import SignLanguageApp  # Lazy import to avoid circular imports
            if self.main_app is None or not self.main_app.isVisible():
                self.main_app = SignLanguageApp()
            self.main_app.showMaximized()
            self.hide()
        except Exception as e:
            print(f"Error launching main app: {e}")
            # Optionally, show a QMessageBox here for user feedback

    def paintEvent(self, event) -> None:
        """Custom paint event for gradient background."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1E1E1E"))
        if self.gradient is None:
            self.gradient = QLinearGradient(0, 0, self.width(), self.height())
            self.gradient.setColorAt(0.0, QColor("#1E1E1E"))
            self.gradient.setColorAt(0.3, QColor("#6B5B95"))
            self.gradient.setColorAt(0.6, QColor("#357ABD"))
            self.gradient.setColorAt(1.0, QColor("#1E1E1E"))
        painter.setOpacity(0.7)
        painter.fillRect(self.rect(), self.gradient)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    home = HomePage()
    home.showMaximized()  # Ensure maximization here
    sys.exit(app.exec())