import sys
import numpy as np
import cv2
import subprocess

frame_width = 640
frame_height = 480

ffmpeg_command = [
    'ffmpeg',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{frame_width}x{frame_height}',
    '-r', '30',
    '-i', '-',
    '-f', 'mpegts',
    'udp://10.246.0.236:1235'
]

# 데이터 읽기
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# 모델 파일들 경로
config_file = "./braille_detect/yolov4-tiny-custom.cfg"
data_file = "./braille_detect/obj.data"
weights_file = "./braille_detect/yolov4-tiny-custom_final.weights"
names_path = "./braille_detect/obj.names"

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


while True:
    raw_frame = sys.stdin.buffer.read(frame_width * frame_height * 3)
    if not raw_frame:
        break
    frame = np.frombuffer(raw_frame, dtype=np.uint8)
    frame = frame.reshape((frame_height, frame_width, 3))

    # ==============================================================================        

    # 모델 처리

    # height, width, channels = frame.shape
    height, width = frame_height, frame_width

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    frame = np.array(frame)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    # 모델에 dot, block 인식해서 마킹

    # ==============================================================================
    
    ffmpeg_process.stdin.write(frame.tobytes())

ffmpeg_process.stdin.close()
ffmpeg_process.wait()
