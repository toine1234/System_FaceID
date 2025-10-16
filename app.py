from flask import Flask, render_template, Response
import cv2
from src.face_detect import FaceDetector

app = Flask(__name__)

detector = FaceDetector("models/yolov8n-face.pt")

def generate_frame():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame,aligned_faces = detector.detect_and_align(frame)

        success, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
      
@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug = True, port=5001)