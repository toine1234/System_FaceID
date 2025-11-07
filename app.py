# app.py - Minified & Optimized Face Recognition Server
import os, sys, time, logging, warnings, cv2, torch, platform
from threading import Thread, Lock, Event
from queue import Queue, Empty
from flask import Flask, render_template, Response, jsonify
from contextlib import contextmanager
from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer
from insightface.app import FaceAnalysis
from src.evaluate import evaluate_system

# ---------------- Config ----------------
CAP_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, CAM_FPS = 0, 640, 480, 20
DETECT_EVERY_N_FRAMES, DETECT_RESIZE, JPEG_QUALITY = 3, (256,192), 75
MAX_QUEUE_SIZE, YOLO_CONF, YOLO_STRIDE = 2, 0.5, 3
PREDICT_IOU, AGNOSTIC_NMS, CUSTOM_NMS_IOU, RECOG_THRESHOLD = 0.4, True, 0.4, 0.6
LOG_INTERVAL, DB_PATH, DATASET_PATH = 5, "encodings/embeddings.pkl", "dataset/SinhVien"
YOLO_MODEL_PATH, MODEL_NAME, DET_SIZE = "models/yolov11n-face.pt", "buffalo_l", (256,256)

# ---------------- App Setup ----------------
app = Flask(__name__)
latest = {"status": "idle"}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceid")
warnings.filterwarnings("ignore")
os.environ.update({"TF_CPP_MIN_LOG_LEVEL":"3","OMP_NUM_THREADS":"1",
"INSIGHTFACE_LOG_LEVEL":"ERROR","ULTRALYTICS_IGNORE_ERRORS":"1"})
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

# ---------------- Models ----------------
device = "cuda" if torch.cuda.is_available() else ("mps" if platform.system()=="Darwin" and torch.backends.mps.is_available() else "cpu")
print(f"[INIT] Loading models on {device.upper()}...")
with suppress_stdout():
    main_face_app = FaceAnalysis(name=MODEL_NAME, allowed_modules=["detection","landmark_3d_68","recognition"], providers=['CPUExecutionProvider'])
    main_face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
    detector = FaceDetector(YOLO_MODEL_PATH, device, main_face_app, DETECT_RESIZE[0], YOLO_CONF, YOLO_STRIDE, PREDICT_IOU, AGNOSTIC_NMS, CUSTOM_NMS_IOU)
    recognizer = FaceRecognizer(device=device, db_path=DB_PATH, threshold=RECOG_THRESHOLD, face_app=main_face_app, detector=detector)
print("[READY] ✅ Models loaded!\n")

# ---------------- Video Capture ----------------
class VideoCaptureThread:
    def __init__(self, src=CAP_DEVICE):
        self.cap = cv2.VideoCapture(src)
        for p,v in [(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH),(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT),(cv2.CAP_PROP_FPS, CAM_FPS)]: self.cap.set(p,v)
        self.lock = Lock()
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.thread = Thread(target=self._update, daemon=True)
    def start(self): self.thread.start(); return self
    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock: self.grabbed, self.frame = grabbed, frame.copy() if grabbed else None
            if not grabbed: time.sleep(0.01)
    def read(self):
        with self.lock: return self.grabbed, self.frame.copy() if self.frame is not None else None
    def stop(self): self.stopped=True; self.thread.join(timeout=1); self.cap.release()

video_capture = VideoCaptureThread().start()

# ---------------- Detection Worker ----------------
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
faces_lock = Lock()
faces_info = []
last_log = {}
stop_event = Event()

def produce_for_detection(f):
    if not frame_queue.full():
        small = cv2.resize(f, DETECT_RESIZE)
        frame_queue.put_nowait((small, f.shape[1], f.shape[0]))

def detection_worker():
    global faces_info, last_log
    while not stop_event.is_set():
        try:
            small, orig_w, orig_h = frame_queue.get(timeout=0.5)
        except Empty: continue
        while not frame_queue.empty(): small, orig_w, orig_h = frame_queue.get_nowait()
        try: _, aligned_faces = detector.detect_and_align(small)
        except Exception as e: logger.error("Detector error: %s", e); aligned_faces=[]
        new_faces = []
        if aligned_faces:
            try: results = recognizer.recognize_faces(aligned_faces)
            except Exception as e: logger.error("Recognizer error: %s", e); results = [("Unknown",0.0)] * len(aligned_faces)
            sx,sy = orig_w/DETECT_RESIZE[0], orig_h/DETECT_RESIZE[1]
            now = time.time()
            for (label,conf), (_, bbox) in zip(results, aligned_faces):
                x1,y1,x2,y2 = bbox
                new_faces.append({"label":label,"conf":float(conf),"bbox":(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)),"ts":now})
        with faces_lock:
            faces_info = new_faces
            for f in faces_info:
                label, conf = f["label"], f["conf"]
                if label != "Unknown" and conf >= RECOG_THRESHOLD and now - last_log.get(label,0) > LOG_INTERVAL:
                    last_log[label] = now
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/attendance_log.csv","a",encoding="utf-8") as log_file:
                        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{label},{conf:.4f}\n")
                    latest.update({"status":"new","name":label,"score":f"{conf*100:.1f}%","time":time.strftime("%H:%M:%S")})

Thread(target=detection_worker, daemon=True).start()

# ---------------- Frame Generator ----------------
def generate_frame():
    fid = 0
    while True:
        grabbed, frame = video_capture.read()
        if not grabbed or frame is None: time.sleep(0.01); continue
        fid += 1
        if fid % DETECT_EVERY_N_FRAMES == 0: produce_for_detection(frame)
        with faces_lock: local_faces = faces_info.copy()
        for f in local_faces:
            x1,y1,x2,y2 = f["bbox"]; label,conf = f["label"],f["conf"]
            color = (0,255,0) if label != "Unknown" else (0,0,255)
            text = f"{label} ({conf*100:.1f}%)" if label != "Unknown" else "Unknown"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,text,(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])
        if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(1 / CAM_FPS)

# ---------------- Flask Routes ----------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/attendance_update")
def attendance_update():
    if latest.get("status") == "new":
        payload = latest.copy()
        latest["status"] = "idle"
        return jsonify(payload)
    return jsonify({"status":"idle"})

# ---------------- Auto Build & Evaluate ----------------
def auto_build_and_evaluate():
    if not os.path.exists(DB_PATH):
        logger.info("[AUTO] Building embeddings DB...")
        temp_recog = FaceRecognizer(device=device, db_path=DB_PATH, threshold=RECOG_THRESHOLD, face_app=main_face_app, detector=detector)
        temp_recog.build_embeddings(DATASET_PATH)
        recognizer._load_db()
    try: evaluate_system()
    except Exception as e: logger.exception("AUTO eval skipped: %s", e)

if __name__ == "__main__":
    try:
        auto_build_and_evaluate()
        logger.info("[RUNNING] Flask FaceID Server started.")
        app.run(debug=False, port=5001, threaded=True)
    except KeyboardInterrupt: logger.info("[MAIN] KeyboardInterrupt")
    finally: stop_event.set(); video_capture.stop()