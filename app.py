from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import platform
import warnings
import os
import logging
import sys
import time
from contextlib import contextmanager

# ================================================================
# ⚙️ Cấu hình Flask & môi trường
# ================================================================
app = Flask(__name__)
latest = {"status": "idle"}  # Dữ liệu điểm danh mới nhất (cho frontend)

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"
os.environ["ULTRALYTICS_IGNORE_ERRORS"] = "1"

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Tắt in ra console tạm thời
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ================================================================
# 1️⃣ Khởi tạo mô hình nhận diện khuôn mặt
# ================================================================
print("\n[KHỞI TẠO] Đang tải mô hình...")

if torch.cuda.is_available():
    device = "cuda"
elif platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"[THIẾT BỊ] Đã chọn: {device.upper()}")

from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer

with suppress_stdout():
    detector = FaceDetector(
        yolo_model_path="models/yolov11n-face.pt",
        device=device
    )
    recognizer = FaceRecognizer(
        device=device,
        db_path="encodings/embeddings.pkl",
        threshold=0.6
    )

print("[SẴN SÀNG] ✅ Mô hình đã được tải thành công!\n")


# ================================================================
# 2️⃣ Xử lý video thời gian thực
# ================================================================
def generate_frame():
    global latest
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        print("❌ Không thể mở webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1️⃣ Phát hiện và căn chỉnh khuôn mặt
        annotated, faces = detector.detect_and_align(frame)

        # 2️⃣ Nhận dạng từng khuôn mặt
        for aligned_face, (x1, y1, x2, y2) in faces:
            try:
                label, score = recognizer.recognize(aligned_face)
                text = f"{label} ({score*100:.1f}%)" if label != "Unknown" else "Unknown"
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                # ✅ Nếu nhận diện thành công (điểm tin cậy >= 0.6)
                if label != "Unknown" and score >= 0.6:
                    latest = {
                        "status": "new",
                        "name": label,
                        "score": f"{score*100:.1f}%",
                        "time": time.strftime("%H:%M:%S")
                    }
                    with open("attendance_log.csv", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{label},{score}\n")


            except Exception:
                continue

        # 3️⃣ Mã hóa khung hình để stream
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ================================================================
# 3️⃣ Định nghĩa các route Flask
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/attendance_update")
def attendance_update():
    """
    Route cho giao diện web lấy thông tin điểm danh mới nhất.
    """
    global latest
    if latest.get("status") == "new":
        payload = latest.copy()
        latest["status"] = "idle"
        return jsonify(payload)
    return jsonify({"status": "idle"})


# ================================================================
# 4️⃣ Chạy Flask server
# ================================================================
if __name__ == '__main__':
    print("[CHẠY] 🚀 Server Flask FaceID đã khởi động")
    print("[THÔNG BÁO] Nhấn CTRL + C để dừng.\n")
    app.run(debug=False, port=5001)
