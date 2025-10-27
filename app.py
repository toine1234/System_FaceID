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
# Flask Configuration & Environment Setup
# ================================================================
app = Flask(__name__)
latest = {"status": "idle"}  # Latest attendance data for frontend

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
os.environ.update({
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "OMP_NUM_THREADS": "1",
    "INSIGHTFACE_LOG_LEVEL": "ERROR",
    "ULTRALYTICS_IGNORE_ERRORS": "1"
})

logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


@contextmanager
def suppress_stdout():
    """Temporarily suppress console output (model loading, etc.)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ================================================================
# 1️ Model Initialization
# ================================================================
print("\n[INIT] Loading models...")

if torch.cuda.is_available():
    device = "cuda"
elif platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"[DEVICE] Selected: {device.upper()}")

from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer

with suppress_stdout():
    detector = FaceDetector(yolo_model_path="models/yolov11n-face.pt", device=device)
    recognizer = FaceRecognizer(device=device, db_path="encodings/embeddings.pkl", threshold=0.6)

print("[READY] ✅ Models loaded successfully!\n")


# ================================================================
# Real-time Video Stream
# ================================================================
def generate_frame():
    """Capture webcam stream, detect & recognize faces in real time."""
    global latest
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ Unable to access webcam.")
        return

    # -------------------------------
    # 🗂️ Tạo thư mục logs nếu chưa có
    # -------------------------------
    os.makedirs("logs", exist_ok=True)
    ATTENDANCE_LOG = os.path.join("logs", "attendance_log.csv")

    # Khởi tạo file nếu chưa có
    if not os.path.exists(ATTENDANCE_LOG):
        open(ATTENDANCE_LOG, "w", encoding="utf-8").write("Datetime,Name,Score\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, faces = detector.detect_and_align(frame)

        for aligned_face, (x1, y1, x2, y2) in faces:
            try:
                label, score = recognizer.recognize(aligned_face)
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                text = f"{label} ({score*100:.1f}%)" if label != "Unknown" else "Unknown"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                if label != "Unknown" and score >= 0.6:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    today = time.strftime("%Y-%m-%d")

                    # 🧾 File 1: Ghi log chi tiết (mọi lần)
                    with open(ATTENDANCE_LOG, "a", encoding="utf-8") as f:
                        f.write(f"{now},{label},{score:.4f}\n")


                    # Cập nhật trạng thái để gửi về frontend
                    latest = {
                        "status": "new",
                        "name": label,
                        "score": f"{score*100:.1f}%",
                        "time": time.strftime("%H:%M:%S")
                    }
            except Exception:
                continue

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()


# ================================================================
# Flask Routes
# ================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/attendance_update")
def attendance_update():
    """Frontend polling route to fetch the latest attendance info."""
    global latest
    if latest.get("status") == "new":
        payload = latest.copy()
        latest["status"] = "idle"
        return jsonify(payload)
    return jsonify({"status": "idle"})


# ================================================================
# Auto Build & Evaluate before starting Flask
# ================================================================
def auto_build_and_evaluate():
    """Tự động huấn luyện & đánh giá hệ thống nếu cần."""
    from src.face_recognize import FaceRecognizer
    from src.evaluate import evaluate_system  # file evaluate.py

    db_path = "encodings/embeddings.pkl"

    # 1️⃣ Nếu chưa có embeddings thì tự động build
    if not os.path.exists(db_path):
        print("[AUTO] 🧠 No embeddings found — building new face database...")
        recognizer = FaceRecognizer(device=device, db_path=db_path, threshold=0.6)
        recognizer.build_embeddings("dataset/SinhVien")
        print("[AUTO] ✅ Embeddings database created.")

    # 2️⃣ Tự động đánh giá độ chính xác
    print("[AUTO] 📊 Evaluating system performance...")
    try:
        evaluate_system()
        print("[AUTO] ✅ Evaluation completed (saved to logs/evaluation_report.csv).")
    except Exception as e:
        print(f"[AUTO] ⚠️ Evaluation skipped: {e}")


# ================================================================
# Launch Flask Server
# ================================================================
if __name__ == "__main__":
    auto_build_and_evaluate()
    print("[RUNNING] 🚀 Flask FaceID Server started.")
    app.run(debug=False, port=5001)
