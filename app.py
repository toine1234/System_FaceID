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
# 1️Model Initialization
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
                    latest = {
                        "status": "new",
                        "name": label,
                        "score": f"{score*100:.1f}%",
                        "time": time.strftime("%H:%M:%S")
                    }
                    with open("attendance_log.csv", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{label},{score:.4f}\n")
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
# Launch Flask Server
# ================================================================
if __name__ == "__main__":
    print("[RUNNING] 🚀 Flask FaceID Server started.")
    print("[INFO] Press CTRL + C to stop.\n")
    app.run(debug=False, port=5001)
