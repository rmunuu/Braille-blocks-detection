import cv2
import subprocess
import numpy as np

# ffmpeg command
command = [
    'ffmpeg',
    '-i', 'udp://0.0.0.0:1235',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '640x480',
    '-'
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

width = 640
height = 480

while True:
    # 데이터 읽기
    raw_frame = process.stdout.read(width * height * 3)
    # if len(raw_frame) != width * height * 3:
    #     break

    # np array로 변환
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

    # display
    cv2.imshow('Received Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

process.stdout.close()
process.wait()
cv2.destroyAllWindows()
