from flask import Flask, render_template, Response
import cv2
from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer

app = Flask(__name__)

detector = FaceDetector("models/yolov8n-face.pt")
recognizer = FaceRecognizer()

#code mới
def generate_frame():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1️⃣ Phát hiện khuôn mặt (YOLO + MTCNN)
        frame, faces = detector.detect_and_align(frame)

        # 2️⃣ Nhận dạng từng khuôn mặt
        for face_img, (x1, y1, x2, y2) in faces:
            try:
                emb = recognizer.get_embedding(face_img)
                label, score = recognizer.recognize_face(
                    emb, "encodings/embeddings.pkl", threshold=0.7
                )

                text = f"{label} ({score:.2f})" if label != "Unknown" else "Unknown"
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                # Vẽ nhãn + khung
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            except Exception as e:
                print(f"[WARN] Nhận dạng khuôn mặt lỗi: {e}")
                continue

        # 3️⃣ Encode và gửi về trình duyệt
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


#code cũ
#    def generate_frame():
#   cap = cv2.VideoCapture(0)
#    while True:
#        success, frame = cap.read()
#       if not success:
#            break

#       frame,aligned_faces = detector.detect_and_align(frame)

#       success, buffer = cv2.imencode('.jpg', frame)
#       frame = buffer.tobytes()
        
#        yield (b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')