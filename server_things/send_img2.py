import cv2
import subprocess

cap = cv2.VideoCapture(0)

ffmpeg_command = [
    'ffmpeg',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '640x480',
    '-r', '30',
    '-i', '-',
    '-f', 'mpegts',
    'udp://10.56.219.242:40001' # 서버 주소, 포트
]

# ffmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ffmpeg_process.stdin.write(frame.tobytes())

cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
