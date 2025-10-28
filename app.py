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
from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer
from insightface.app import FaceAnalysis
from src.evaluate import evaluate_system


# ================================================================
# Flask Configuration & Environment Setup
# ================================================================
app = Flask(__name__)
latest = {"status": "idle"}

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

with suppress_stdout():
    
    # 1. Xác định provider cho ONNX (InsightFace)
    print("[INIT] Detecting ONNX providers...")
    if device == "cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0 # Dùng GPU đầu tiên
    elif device == "mps":
        # Ưu tiên CoreML cho Apple Silicon (nhanh nhất)
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0 # Dùng GPU (CoreML)
        print("[INFO] Using CoreMLExecutionProvider for InsightFace.")
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1 # Dùng CPU
        print("[INFO] Using CPUExecutionProvider for InsightFace.")

    # 2. TẠO 1 FaceAnalysis DUY NHẤT
    print("[INIT] Loading Consolidated FaceAnalysis (buffalo_l)...")
    main_face_app = FaceAnalysis(name="buffalo_l", 
                                 allowed_modules=["detection", "landmark_3d_68", "recognition"],
                                 providers=providers) # <-- TRUYỀN PROVIDERS VÀO
    
    main_face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))
    print("[INIT] FaceAnalysis consolidated.")

    # 3. Truyền main_face_app vào Detector
    detector = FaceDetector(
        yolo_model_path="models/yolov11n-face.pt", 
        device=device,          # YOLO sẽ dùng 'mps'
        face_app=main_face_app, # InsightFace sẽ dùng 'CoreML'
        yolo_stride=4           # <-- Tăng stride lên 4 để mượt hơn
    )
    
    # 4. Truyền cả main_face_app VÀ detector vào Recognizer
    recognizer = FaceRecognizer(
        device=device, 
        db_path="encodings/embeddings.pkl", 
        threshold=0.6,
        detector=detector,
        face_app=main_face_app
    )
    
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

    os.makedirs("logs", exist_ok=True)
    ATTENDANCE_LOG = os.path.join("logs", "attendance_log.csv")
    if not os.path.exists(ATTENDANCE_LOG):
        open(ATTENDANCE_LOG, "w", encoding="utf-8").write("Datetime,Name,Score\n")
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect và Align (trả về list các khuôn mặt)
        annotated, faces = detector.detect_and_align(frame)
        
        if faces:
            try:
                # 2. Gọi hàm nhận diện THEO LÔ (BATCH) 1 LẦN DUY NHẤT
                results = recognizer.recognize_faces(faces) # results là List[Tuple[label, score]]

                # 3. Lặp qua kết quả và dữ liệu khuôn mặt ĐÃ CÓ
                for (label, score), (aligned_img, (x1, y1, x2, y2)) in zip(results, faces):
                    
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    text = f"{label} ({score*100:.1f}%)" if label != "Unknown" else "Unknown"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                    if label != "Unknown" and score >= 0.6:
                        now = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Ghi log
                        with open(ATTENDANCE_LOG, "a", encoding="utf-8") as f:
                            f.write(f"{now},{label},{score:.4f}\n")

                        # Cập nhật trạng thái
                        latest = {
                            "status": "new",
                            "name": label,
                            "score": f"{score*100:.1f}%",
                            "time": time.strftime("%H:%M:%S")
                        }
            except Exception as e:
                logging.error(f"Recognition batch failed: {e}")
                pass # Bỏ qua frame này nếu có lỗi
        # ================================================================

        # 4. Gửi frame về trình duyệt
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
    db_path = "encodings/embeddings.pkl"

    # Nếu chưa có embeddings thì tự động build
    if not os.path.exists(db_path):
        print("[AUTO] 🧠 No embeddings found — building new face database...")
        
        temp_recognizer = FaceRecognizer(
            device=device, 
            db_path=db_path, 
            threshold=0.6,
            detector=detector,
            face_app=main_face_app
        )
        temp_recognizer.build_embeddings("dataset/SinhVien")
        print("[AUTO] ✅ Embeddings database created.")
        
        # Cập nhật lại DB cho recognizer chính
        print("[AUTO] Reloading DB for main recognizer...")
        recognizer._load_db()

    # Tự động đánh giá độ chính xác
    print("[AUTO] 📊 Evaluating system performance...")
    try:
        # Truyền recognizer đã được tải đầy đủ vào
        evaluate_system() 
        print("[AUTO] ✅ Evaluation completed (saved to logs/evaluation_report.csv).")
    except Exception as e:
        print(f"[AUTO] ⚠️ Evaluation skipped: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    auto_build_and_evaluate()
    print("[RUNNING] 🚀 Flask FaceID Server started.")
    app.run(debug=False, port=5001)