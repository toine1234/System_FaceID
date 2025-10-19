"""
app.py - Flask API realtime nhận dạng khuôn mặt
Tác giả: Thanh Trúc (2025)
"""

from flask import Flask, render_template, Response
import cv2
from src.f import FaceDetector
from src.face_recognize import FaceRecognizer

# ================================================================
# 1️⃣ Khởi tạo Flask app
# ================================================================
app = Flask(__name__)

# ================================================================
# 2️⃣ Khởi tạo mô hình
# ================================================================
print("[INIT] Loading models...")

detector = FaceDetector(
    yolo_model_path="models/yolov11n-face.pt",
    device="mps"
)
recognizer = FaceRecognizer(
    device="mps",
    pretrained_model="vggface2",
    embeddings_path="encodings/embeddings.pkl"
)

print("[READY] Models loaded successfully!")

# ================================================================
# 3️⃣ Hàm xử lý luồng video
# ================================================================
def generate_frame():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1️⃣ Phát hiện + căn chỉnh khuôn mặt
        frame, faces = detector.detect_and_align(frame)

        # 2️⃣ Nhận dạng từng khuôn mặt
        for face_img, (x1, y1, x2, y2) in faces:
            try:
                emb = recognizer.get_embedding(face_img)
                label, score = recognizer.recognize_face(emb, threshold=0.7)

                text = f"{label} ({score*100:.1f}%)" if label != "Unknown" else "Unknown"
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                # Vẽ khung + nhãn
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            except Exception as e:
                print(f"[WARN] Lỗi nhận dạng: {e}")
                continue

        # 3️⃣ Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ================================================================
# 4️⃣ Flask Routes
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================================================
# 5️⃣ Main
# ================================================================
if __name__ == '__main__':
    print("[RUNNING] Flask FaceID server started at http://127.0.0.1:5001/")
    app.run(debug=True, port=5001)
