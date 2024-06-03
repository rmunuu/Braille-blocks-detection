import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QDesktopWidget, QHBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tmap")

        # 컴퓨터 스크린 전체 사이즈 가져오기
        screen = QDesktopWidget().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.browser = QWebEngineView()
        self.browser.setFixedWidth(int(self.screen_width * 0.9)) # 왼쪽 90% 만큼 크기
        self.layout.addWidget(self.browser)

        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        # tmap api html 파일 로드
        self.browser.setUrl(QUrl.fromLocalFile("C:/Users/User/Desktop/학교/_3-1/컴과프/project/map.html"))

        # Start Camera 버튼 생성
        self.start_button = QPushButton('Start Camera', self)
        self.start_button.clicked.connect(self.start_camera)
        self.right_layout.addWidget(self.start_button)

        # 카메라 레이블
        self.camera_label = QLabel(self)
        self.right_layout.addWidget(self.camera_label)
        
        # 카메라, 타이머 초기화
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.cap is None:
            # 시작 화면 숨기기
            self.browser.hide()
            self.start_button.hide()

            # 카메라 초기화
            self.cap = cv2.VideoCapture(0)

            # 카메라 피드 업데이트 위한 타이머 시작
            self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, aspectRatioMode = 1)
            self.camera_label.setPixmap(QPixmap.fromImage(p))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.cap is None:
                self.close()  # 초기상태에 있다면 창 닫기
            else:
                self.stop_camera()

    def stop_camera(self):
        if self.cap is not None:
            # 타이머 멈추고 카메라 닫기
            self.timer.stop()
            self.cap.release()
            self.cap = None
            # 시작화면 띄우기
            self.browser.show()
            self.start_button.show()
            self.camera_label.clear()

    def closeEvent(self, event):
        self.stop_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec_())
