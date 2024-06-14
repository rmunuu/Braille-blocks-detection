import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QDesktopWidget,
    QHBoxLayout,
    QDialog,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 모델 불러오기
model_cur = torch.hub.load(
    "./yolov5",
    "custom",
    path="./check/best_20240609_0909.pt",
    source="local",
    force_reload=True,
)

green_cnt, cnt = 0, 0  # 긴 선분의 개수 세는 것
flag_l, flag_r = False, False  # 좌우 선분에서

# 불러올 데이터
config_file = "./check/yolov4-tiny-custom.cfg"
data_file = "./check/obj.data"
weights_file = "./check/yolov4-tiny-custom_final.weights"
output_path = "output_image.jpg"
names_path = "./check/obj.names"

# 모델 불러오기
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except cv2.error as e:
    print(f"Error loading YOLO files: {e}")
    exit()

try:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except cv2.error as e:
    print(f"Error getting layer names: {e}")
    exit()

try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: obj.names file not found")
    exit()


# 관심영역 채우기
def region_of_interest(img, vertices):
    # 이 함수의 경우에는 도움 받음 - gpt, 마스크 처리한 이미지 반환하는 함수
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# 선 그리기
def draw_the_lines(img, lines):
    imge = np.copy(img)
    global green_cnt, flag_l, flag_r, cnt
    green_cnt = 0
    cnt = 0  # green_cnt 초기화
    flag_r, flag_l = False, False  # flag들 초기화
    blank_image = np.zeros((imge.shape[0], imge.shape[1], 3), dtype=np.uint8)
    if lines is None:
        return imge

    # line들에 대해서 길이를 측정하고 그리기, 길이에 따라 유형 구분
    for line in lines:
        for x1, y1, x2, y2 in line:
            # print(x1,x2,y1,y2)
            color = (255, 0, 0)
            cnt += 1
            # 꺽이는지 여부에 따라서 다른 색을 표현하고자 함
            if abs(x1 - x2) / img.shape[1] < abs(y1 - y2) / img.shape[0]:
                color = (0, 255, 0)
                green_cnt += 1
                if x1 < img.shape[1] // 2 and x2 < img.shape[1] * 2 // 3:
                    flag_l = True
                elif x1 > img.shape[1] // 2 and x1 > img.shape[1] // 3:
                    flag_r = True
            cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness=5)
            imge = cv2.addWeighted(imge, 0.8, blank_image, 0.1, 0.0)
    return imge


# 이미지 경계 검출 처리하기
def process_image(image):
    global green_cnt, flag_r, flag_l, cnt
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[0], image.shape[1]
    region_of_interest_coor = [
        (0, height // 2),
        (width, height // 2),
        (0, height),
        (width, height),
    ]

    # 이미지에 필터 처리하여서 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # blackhat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, k)
    eroded = cv2.erode(image, k)
    gray_image = cv2.cvtColor(eroded, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
    # blurred = cv2.Laplacian(gray_image,-1)
    canny_image = cv2.Canny(blurred, 50, 200)

    # gpt의 도움 받음 - roi 이미지 선택하고 이에서 허프 변환을 통해 선을 탐지
    cropped = region_of_interest(
        canny_image, np.array([region_of_interest_coor], np.int32)
    )
    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=100,
    )
    image_with_lines = draw_the_lines(image, lines)
    # 여기까지만

    # 이미지 합치고 green_cnt의 개수에 따라서 위험 정도 판별
    # 긴 경계의 수가 많으면 그만큼 경계로 둘러싸일 확률이 높다 -> 안전할 확률이 높다
    # print(green_cnt, cnt, flag_l, flag_r)
    combined_image = cv2.addWeighted(image, 0.8, image_with_lines, 1, 1)
    if green_cnt / (cnt + 1) < 0.2 and (flag_l == False and flag_r == False):
        cv2.putText(
            combined_image,
            "Danger!",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            3,
        )
    elif green_cnt / (cnt + 1) < 0.5 or (flag_l == False or flag_r == False):
        cv2.putText(
            combined_image,
            "Maybe Danger!",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            3,
        )
    else:
        cv2.putText(
            combined_image,
            "No Bangsim!",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            3,
        )
    return combined_image


def check_braile(frame):
    global net
    ret = frame.copy()
    height, width, channels = frame.shape
    cnt_dot = 0
    # Detecting objects
    blob = cv2.dnn.blobFromImage(ret, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "dot":
                cnt_dot += 1
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(ret, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                ret,
                label + " " + str(round(confidence, 2)),
                (x, y + 30),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                color,
                3,
            )
    if cnt_dot > 0:
        cv2.putText(ret, "yes dot",(0,60),cv2.FONT_HERSHEY_PLAIN,6,(0,255,0),3)
    else:
        cv2.putText(ret, "no dot",(0,60),cv2.FONT_HERSHEY_PLAIN,6,(0,255,0),3)
    return ret


class CameraWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Camera Window")
        self.setGeometry(0, 0, parent.screen_width, parent.screen_height)

        self.layout = QVBoxLayout(self)

        self.camera_label = QLabel(self)
        self.layout.addWidget(self.camera_label)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture("./check/v4.mp4")
            self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            move_cnt = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = frame.copy()
            result1 = model_cur(frame1)
            labels = result1.names
            for _, det in enumerate(result1.xyxy[0]):
                xmin, ymin, xmax, ymax, conf, cls = det
                if labels[int(cls)] == "movable_signage":
                    move_cnt += 1
            ret_frame1 = np.squeeze(result1.render())
            cv2.putText(
                ret_frame1,
                "Count of moving things!" + " " + str(move_cnt),
                (0, ret_frame1.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            ret_frame1 = cv2.resize(
                ret_frame1, (ret_frame1.shape[1] // 3, ret_frame1.shape[0] // 3)
            )
            frame2 = frame.copy()
            ret_frame2 = process_image(frame2)
            ret_frame2 = cv2.cvtColor(ret_frame2, cv2.COLOR_RGB2BGR)
            ret_frame2 = cv2.resize(
                ret_frame2, (ret_frame2.shape[1] // 3, ret_frame2.shape[0] // 3)
            )
            frame3 = frame.copy()
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR)
            ret_frame3 = check_braile(frame3)
            ret_frame3 = cv2.resize(
                ret_frame3, (ret_frame3.shape[1] // 3, ret_frame3.shape[0] // 3)
            )
            ret_frame3 = cv2.cvtColor(ret_frame3, cv2.COLOR_BGR2RGB)
            last_frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
            combined_frame1 = np.hstack((ret_frame1, ret_frame2))
            combined_frame2 = np.hstack((ret_frame3, last_frame))
            combined_frame = np.vstack((combined_frame1, combined_frame2))
            combined_frame = cv2.resize(
                combined_frame, (self.width(), self.height())
            )
            h, w, ch = combined_frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                combined_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            p = convert_to_Qt_format.scaled(1280, 960, aspectRatioMode=0)
            self.camera_label.setPixmap(QPixmap.fromImage(p))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_camera()
            self.close()

    def stop_camera(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_label.clear()


class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tmap")
        screen = QDesktopWidget().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.browser = QWebEngineView()
        self.browser.setFixedWidth(int(self.screen_width * 0.9))
        self.layout.addWidget(self.browser)

        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        self.browser.setUrl(QUrl.fromLocalFile("C:\project_opencv\map.html"))

        self.start_button = QPushButton("Start Camera", self)
        self.start_button.clicked.connect(self.open_camera_window)
        self.right_layout.addWidget(self.start_button)

    def open_camera_window(self):
        self.camera_window = CameraWindow(self)
        self.camera_window.start_camera()
        self.camera_window.show()

    def closeEvent(self, event):
        if hasattr(self, 'camera_window') and self.camera_window.isVisible():
            self.camera_window.stop_camera()
            self.camera_window.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec_())


'''
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
        self.browser.setFixedWidth(int(self.screen_width * 0.9))  # 왼쪽 90% 만큼 크기
        self.layout.addWidget(self.browser)

        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        # tmap api html 파일 로드
        # self.browser.setUrl(QUrl.fromLocalFile("C:\project_opencv\index.html"))
        self.browser.load(QtCore.QUrl.fromLocalFile(QtCore.QDir.current().filePath("C:\project_opencv\index.html")))


        # Start Camera 버튼 생성
        self.start_button = QPushButton("Start Camera", self)
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
            self.cap = cv2.VideoCapture("./check/video2.mp4")

            # 카메라 피드 업데이트 위한 타이머 시작
            self.timer.start(20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            move_cnt = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame1 = frame.copy()
            result1 = model_cur(frame1)
            labels = result1.names
            for _, det in enumerate(result1.xyxy[0]):  # 이미지 내에서 감지된 것
                xmin, ymin, xmax, ymax, conf, cls = det
                if labels[int(cls)] == "movable_signage":  # 이동하는 것 확인
                    move_cnt += 1
            ret_frame1 = np.squeeze(result1.render())
            cv2.putText(
                ret_frame1,
                "Count of moving things!" + " " + str(move_cnt),
                (0, ret_frame1.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # 나머지 코드들로 처리한 이미지 병합 및 띄우기
            ret_frame1 = cv2.resize(
                ret_frame1, (ret_frame1.shape[1] // 3, ret_frame1.shape[0] // 3)
            )
            frame2 = frame.copy()
            ret_frame2 = process_image(frame2)
            ret_frame2 = cv2.cvtColor(ret_frame2, cv2.COLOR_RGB2BGR)
            ret_frame2 = cv2.resize(
                ret_frame2, (ret_frame2.shape[1] // 3, ret_frame2.shape[0] // 3)
            )
            frame3 = frame.copy()
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR)
            ret_frame3 = check_braile(frame3)
            ret_frame3 = cv2.resize(
                ret_frame3, (ret_frame3.shape[1] // 3, ret_frame3.shape[0] // 3)
            )
            ret_frame3 = cv2.cvtColor(ret_frame3, cv2.COLOR_BGR2RGB)
            last_frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
            combined_frame1 = np.hstack((ret_frame1, ret_frame2))
            combined_frame2 = np.hstack((ret_frame3, last_frame))
            combined_frame = np.vstack((combined_frame1, combined_frame2))
            combined_frame = cv2.resize(
                combined_frame, (self.screen_width, self.screen_height)
            )
            # Qimage를 통한 처리
            h, w, ch = combined_frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                combined_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            p = convert_to_Qt_format.scaled(1280, 960, aspectRatioMode=0)
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
'''