import sys
from PyQt6.QtWidgets import QApplication
from HomePage import HomePage

def main():
    app = QApplication(sys.argv)
    home = HomePage()
    home.showMaximized()  # Ensure maximization
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
