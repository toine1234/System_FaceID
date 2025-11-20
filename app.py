import os, sys, time, logging, warnings, cv2, torch, platform
from threading import Thread, Lock, Event
from queue import Queue, Empty
from flask import Flask, render_template, Response, jsonify
from contextlib import contextmanager
from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer
from insightface.app import FaceAnalysis

CONFIG = {
    "CAP_DEVICE": 0, "FRAME_WIDTH": 640, "FRAME_HEIGHT": 480, "CAM_FPS": 20,
    "DETECT_EVERY_N_FRAMES": 3, "DETECT_RESIZE": (256, 192), "JPEG_QUALITY": 75,
    "MAX_QUEUE_SIZE": 2, "YOLO_CONF": 0.5, "YOLO_STRIDE": 3,
    "PREDICT_IOU": 0.4, "AGNOSTIC_NMS": True, "CUSTOM_NMS_IOU": 0.4, "RECOG_THRESHOLD": 0.6,
    "LOG_INTERVAL": 5, "EMBEDDINGS_DIR": "encodings/", "DATASET_PATH": "dataset/SinhVien",
    "YOLO_MODEL_PATH": "models/yolov11n-face.pt", "MODEL_NAME": "buffalo_l", "DET_SIZE": (256, 256)
}

app = Flask(__name__)
latest = {"status": "idle"}

# GLOBAL FPS
fps_info = {"last_time": time.time(), "fps": 0.0, "frame_count": 0}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceid")
warnings.filterwarnings("ignore")
os.environ.update({"TF_CPP_MIN_LOG_LEVEL":"3", "OMP_NUM_THREADS":"1",
                   "INSIGHTFACE_LOG_LEVEL":"ERROR", "ULTRALYTICS_IGNORE_ERRORS":"1"})
for name in ["insightface", "onnxruntime"]:
    logging.getLogger(name).setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

device = "cuda" if torch.cuda.is_available() else ("mps" if platform.system()=="Darwin" and torch.backends.mps.is_available() else "cpu")
print(f"[INIT] Loading models on {device.upper()}...")
with suppress_stdout():
    main_face_app = FaceAnalysis(name=CONFIG["MODEL_NAME"], allowed_modules=["detection","landmark_3d_68","recognition"], providers=['CPUExecutionProvider'])
    main_face_app.prepare(ctx_id=-1, det_size=CONFIG["DET_SIZE"])
    detector = FaceDetector(CONFIG["YOLO_MODEL_PATH"], device, main_face_app, CONFIG["DETECT_RESIZE"][0], 
                           CONFIG["YOLO_CONF"], CONFIG["YOLO_STRIDE"], CONFIG["PREDICT_IOU"], CONFIG["AGNOSTIC_NMS"], CONFIG["CUSTOM_NMS_IOU"])
    recognizer = FaceRecognizer(device=device, db_path=None, embeddings_dir=CONFIG["EMBEDDINGS_DIR"], 
                               threshold=CONFIG["RECOG_THRESHOLD"], face_app=main_face_app, detector=detector)
print("[READY] ✅ Models loaded!\n")

class VideoCaptureThread:
    def __init__(self, src=CONFIG["CAP_DEVICE"]):
        self.cap = cv2.VideoCapture(src)
        for p, v in [(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"]), 
                     (cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"]), 
                     (cv2.CAP_PROP_FPS, CONFIG["CAM_FPS"])]:
            self.cap.set(p, v)
        self.lock = Lock()
        self.frame, self.grabbed, self.stopped = None, False, False
        self.thread = Thread(target=self._update, daemon=True)
    
    # 1. Xác định provider cho ONNX (InsightFace)
    print("[INIT] Detecting ONNX providers...")
    if device == "cuda":
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
    elif device == "mps":
        providers = ['CPUExecutionProvider']
        ctx_id = 0
        print("[INFO] Using CoreMLExecutionProvider for InsightFace.")
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1
        print("[INFO] Using CPUExecutionProvider for InsightFace.")

    # 2. TẠO 1 FaceAnalysis DUY NHẤT
    print("[INIT] Loading Consolidated FaceAnalysis (buffalo_l)...")
    main_face_app = FaceAnalysis(name="buffalo_l", 
                                 allowed_modules=["detection", "landmark_3d_68", "recognition"],
                                 providers=providers)
    def start(self):
        self.thread.start()
        return self
    
    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame.copy() if grabbed else None
            if not grabbed:
                time.sleep(0.01)
    
    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()

video_capture = VideoCaptureThread().start()

frame_queue = Queue(maxsize=CONFIG["MAX_QUEUE_SIZE"])
faces_lock, faces_info, last_log, stop_event = Lock(), [], {}, Event()

def produce_for_detection(f):
    if not frame_queue.full():
        frame_queue.put_nowait((cv2.resize(f, CONFIG["DETECT_RESIZE"]), f.shape[1], f.shape[0]))

def detection_worker():
    global faces_info, last_log
    while not stop_event.is_set():
        try:
            small, orig_w, orig_h = frame_queue.get(timeout=0.5)
        except Empty:
            continue
        
        while not frame_queue.empty():
            small, orig_w, orig_h = frame_queue.get_nowait()
        
        try:
            _, aligned_faces = detector.detect_and_align(small)
            results = recognizer.recognize_faces(aligned_faces) if aligned_faces else []
        except Exception as e:
            logger.error("Detection error: %s", e)
            aligned_faces, results = [], []
        
        sx, sy = orig_w / CONFIG["DETECT_RESIZE"][0], orig_h / CONFIG["DETECT_RESIZE"][1]
        now, new_faces = time.time(), []
        
        for (label, conf), (_, bbox) in zip(results, aligned_faces):
            x1, y1, x2, y2 = bbox
            new_faces.append({"label": label, "conf": float(conf), "bbox": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)), "ts": now})
        
        with faces_lock:
            faces_info = new_faces
            for f in faces_info:
                label, conf = f["label"], f["conf"]
                if label != "Unknown" and conf >= CONFIG["RECOG_THRESHOLD"] and now - last_log.get(label, 0) > CONFIG["LOG_INTERVAL"]:
                    last_log[label] = now
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/attendance_log.csv", "a", encoding="utf-8") as log:
                        fps_value = fps_info["fps"]
                        log.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                            f"{label},{conf:.4f},{fps_value:.2f}\n"
                        )
                    latest.update({"status": "new", "name": label, "score": f"{conf*100:.1f}%", "time": time.strftime("%H:%M:%S")})

Thread(target=detection_worker, daemon=True).start()

def generate_frame():
    fid = 0
    while True:
        grabbed, frame = video_capture.read()

        # ==== FPS TRACKING ====
        fps_info["frame_count"] += 1
        now = time.time()
        elapsed = now - fps_info["last_time"]

        # Cập nhật FPS mỗi 1 giây
        if elapsed >= 1.0:
            fps_info["fps"] = fps_info["frame_count"] / elapsed
            fps_info["frame_count"] = 0
            fps_info["last_time"] = now

        # Vẽ FPS lên frame
        cv2.putText(frame, f"FPS: {fps_info['fps']:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 80), 2, cv2.LINE_AA)
        
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue
        
        fid += 1
        if fid % CONFIG["DETECT_EVERY_N_FRAMES"] == 0:
            produce_for_detection(frame)
        
        with faces_lock:
            local_faces = faces_info.copy()
        
        for f in local_faces:
            x1, y1, x2, y2 = f["bbox"]
            label, conf = f["label"], f["conf"]
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            text = f"{label} ({conf*100:.1f}%)" if label != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG["JPEG_QUALITY"]])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(1 / CONFIG["CAM_FPS"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/attendance_update")
def attendance_update():
    if latest.get("status") == "new":
        payload = latest.copy()
        latest["status"] = "idle"
        return jsonify(payload)
    return jsonify({"status": "idle"})

# def auto_build_and_evaluate():
#     has_emb = os.path.exists(CONFIG["EMBEDDINGS_DIR"]) and any(
#         f.endswith("_embedding.pkl") for f in os.listdir(CONFIG["EMBEDDINGS_DIR"])
#     )
    
#     if not has_emb:
#         logger.info("[AUTO] Building embeddings...")
#         temp = FaceRecognizer(device=device, db_path=None, embeddings_dir=CONFIG["EMBEDDINGS_DIR"],
#                              threshold=CONFIG["RECOG_THRESHOLD"], face_app=main_face_app, detector=detector)
#         temp.build_embeddings(CONFIG["DATASET_PATH"], train_classifier=True)
#         recognizer._load_db()

if __name__ == "__main__":
    try:
        # auto_build_and_evaluate()
        logger.info("[RUNNING] Flask server on port 5001")
        app.run(debug=False, port=5001, threaded=True)
    except KeyboardInterrupt:
        logger.info("[MAIN] KeyboardInterrupt")
    finally:
        stop_event.set()
        video_capture.stop()
