import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# app = QApplication(sys.argv)
# screen = QDesktopWidget().screenGeometry()
# w = screen.width()
# h = screen.height()
# app.quit()

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tmap")
        # self.setGeometry(0, 0, w, h)
        self.setGeometry(100, 100, 1200, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.browser = QWebEngineView()
        self.layout.addWidget(self.browser)

        # Load the local HTML file
        self.browser.setUrl(QUrl.fromLocalFile("C:/Users/User/Desktop/학교/_3-1/컴과프/project/map.html"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec_())
