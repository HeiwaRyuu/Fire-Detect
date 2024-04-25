from flask import Flask, Response, request
import os
import cv2
import time

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

video_paths = [os.path.join(current_dir, "..", "videos", f) for f in os.listdir(os.path.join(current_dir, "..", "videos"))]

frame_cache = {}

def generate_frames(video_path):
    if video_path not in frame_cache:
        video = cv2.VideoCapture(video_path)
        frame_cache[video_path] = video

    video = frame_cache[video_path]

    while True:
        ret, frame = video.read()

        # Verificar se o vídeo terminou
        if not ret:
            video.release()
            frame_cache.pop(video_path, None)
            break

        frame = cv2.resize(frame, (320, 240))

        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            break

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    video_index = int(request.args.get('video_index', 0))
    if video_index < 0 or video_index >= len(video_paths):
        return "Invalid video index", 400
    return Response(generate_frames(video_paths[video_index]), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
