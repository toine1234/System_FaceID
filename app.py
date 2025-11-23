# app_v7.py - Updated for AntiSpoofV7
import os
import sys
import time
import logging
import warnings
import cv2
import torch
import platform

from threading import Thread, Lock, Event
from queue import Queue, Empty
from flask import Flask, render_template, Response, jsonify
from contextlib import contextmanager

from src.face_detect import FaceDetector
from src.face_recognize import FaceRecognizer
from src.anti_spoof_v6 import AntiSpoofV7
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
CONFIG = {
    "CAP_DEVICE": 0, "FRAME_WIDTH": 640, "FRAME_HEIGHT": 480, "CAM_FPS": 20,
    "DETECT_EVERY_N_FRAMES": 3, "DETECT_RESIZE": (256, 192), "JPEG_QUALITY": 75,
    "MAX_QUEUE_SIZE": 2,
    "YOLO_MODEL_PATH": "models/yolov11n-face.pt",
    "MODEL_NAME": "buffalo_l",
    "DET_SIZE": (256, 256),
    "RECOG_THRESHOLD": 0.6,
    "LOG_INTERVAL": 5,
    "EMBEDDINGS_DIR": "encodings/",
    "DATASET_PATH": "dataset/SinhVien"
}
EXPECTED_ID = "2033225652"

app = Flask(__name__)
latest = {"status": "idle"}

# GLOBAL FPS
fps_info = {"last_time": time.time(), "fps": 0.0, "frame_count": 0}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faceid")
warnings.filterwarnings("ignore")
os.environ.update({"TF_CPP_MIN_LOG_LEVEL": "3", "OMP_NUM_THREADS": "1"})

device = "cuda" if torch.cuda.is_available() else ("mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu")
print(f"[INIT] Loading models on {device.upper()}...")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ---------------- Load models ----------------
with suppress_stdout():
    main_face_app = FaceAnalysis(
        name=CONFIG["MODEL_NAME"],
        allowed_modules=["detection", "landmark_3d_68", "recognition"],
        providers=['CPUExecutionProvider']
    )
    main_face_app.prepare(ctx_id=-1, det_size=CONFIG["DET_SIZE"])

    detector = FaceDetector(
        CONFIG["YOLO_MODEL_PATH"], device, main_face_app,
        CONFIG["DETECT_RESIZE"][0], 0.5, 3, 0.4, True, 0.4
    )
    recognizer = FaceRecognizer(
        device=device, db_path=None, embeddings_dir=CONFIG["EMBEDDINGS_DIR"],
        threshold=CONFIG["RECOG_THRESHOLD"], face_app=main_face_app, detector=detector
    )

# ---------------- INIT AntiSpoofV7 ----------------
df_detector = AntiSpoofV7(
    face_size=(224,224),
    buffer_size=8,
    blink_thr=0.15,
    motion_thr=0.08,
    fft_thr=0.12,
    moire_thr=15.0,
    spoof_score_limit=0.55
)
print("[READY] ✅ Models loaded!\n")

# ---------------- Video Capture Thread ----------------
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

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame.copy() if grabbed else None
            if not grabbed: time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()

video_capture = VideoCaptureThread().start()

# ---------------- Worker / Queue ----------------
frame_queue = Queue(maxsize=CONFIG["MAX_QUEUE_SIZE"])
faces_lock, faces_info, last_log, stop_event = Lock(), [], {}, Event()

def produce_for_detection(frame):
    if not frame_queue.full():
        small = cv2.resize(frame, CONFIG["DETECT_RESIZE"])
        frame_queue.put_nowait((small, frame.shape[1], frame.shape[0]))

def detection_worker():
    global faces_info, last_log
    while not stop_event.is_set():
        try:
            small, orig_w, orig_h = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        while not frame_queue.empty():
            try:
                small, orig_w, orig_h = frame_queue.get_nowait()
            except Empty:
                break

        try:
            det_out = detector.detect_and_align(small)
            aligned_faces, landmarks_list = [], []
            if isinstance(det_out, (tuple, list)):
                if len(det_out) == 2:
                    _, af = det_out
                    aligned_faces = af or []
                    landmarks_list = [None] * len(aligned_faces)
                elif len(det_out) >= 3:
                    _, af, lm_list = det_out[:3]
                    aligned_faces = af or []
                    landmarks_list = lm_list or [None] * len(aligned_faces)
        except Exception as e:
            logger.error("Detection error: %s", e)
            aligned_faces, landmarks_list = [], []

        sx = orig_w / CONFIG["DETECT_RESIZE"][0]
        sy = orig_h / CONFIG["DETECT_RESIZE"][1]
        now = time.time()
        new_faces = []

        N = min(len(aligned_faces), len(aligned_faces))
        for i in range(N):
            try:
                af = aligned_faces[i]
                _, small_bbox = af[0], tuple(af[1])
                lmk = landmarks_list[i] if (landmarks_list and i < len(landmarks_list)) else None
                label, conf = recognizer.recognize_faces([af])[0] if recognizer else ("Unknown",0.0)

                x1_s, y1_s, x2_s, y2_s = small_bbox
                x1_o, y1_o, x2_o, y2_o = int(x1_s*sx), int(y1_s*sy), int(x2_s*sx), int(y2_s*sy)

                # ---------------- DeepFake Check ----------------
                is_fake, fake_conf = df_detector.check_frame(small, small_bbox, landmarks=lmk, face_id=(label if label!="Unknown" else None))

                new_faces.append({
                    "label": label, "conf": float(conf),
                    "bbox_small": (int(x1_s), int(y1_s), int(x2_s), int(y2_s)),
                    "bbox": (x1_o, y1_o, x2_o, y2_o),
                    "ts": now, "landmarks": lmk,
                    "is_fake": is_fake, "fake_conf": fake_conf
                })

            except Exception as e:
                continue

        with faces_lock:
            faces_info = new_faces
            # Logging
            for f in faces_info:
                label = f["label"]
                if not f["is_fake"] and now - last_log.get(label,0)>CONFIG["LOG_INTERVAL"]:
                    last_log[label] = now
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/attendance_log.csv","a",encoding="utf-8") as log:
                        fps_value = fps_info["fps"]
                        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{EXPECTED_ID},{label},{f['conf']:.4f},{fps_value:.2f}\n")
                    latest.update({"status":"new","name":label,"score":f"{f['conf']*100:.1f}%","time":time.strftime("%H:%M:%S")})

Thread(target=detection_worker, daemon=True).start()

# ---------------- Frame Generator ----------------
def generate_frame():
    fid = 0
    while True:
        grabbed, frame = video_capture.read()
        fps_info["frame_count"] += 1
        now = time.time()
        if now - fps_info["last_time"] >= 1.0:
            fps_info["fps"] = fps_info["frame_count"] / (now - fps_info["last_time"])
            fps_info["frame_count"] = 0
            fps_info["last_time"] = now

        if frame is None: time.sleep(0.01); continue
        fid += 1
        if fid % CONFIG["DETECT_EVERY_N_FRAMES"] == 0:
            produce_for_detection(frame)

        with faces_lock:
            local_faces = list(faces_info)

        for f in local_faces:
            x1, y1, x2, y2 = f["bbox"]
            label, conf = f["label"], f["conf"]
            is_fake = f.get("is_fake", False)
            fake_conf = f.get("fake_conf", 0.0)
            color = (0,255,0) if not is_fake else (0,0,255)
            text = f"{label} ({int(conf*100)}%)" if not is_fake else f"FAKE ({int(fake_conf*100)}%)"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,text,(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        ok, buffer = cv2.imencode(".jpg", frame,[cv2.IMWRITE_JPEG_QUALITY, CONFIG["JPEG_QUALITY"]])
        if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buffer.tobytes()+b"\r\n"
        time.sleep(1/CONFIG["CAM_FPS"])

# ---------------- Flask Routes ----------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/attendance_update")
def attendance_update():
    if latest.get("status")=="new":
        payload = latest.copy()
        latest["status"]="idle"
        return jsonify(payload)
    return jsonify({"status":"idle"})

if __name__=="__main__":
    try:
        logger.info("[RUNNING] Flask server on port 5001")
        app.run(debug=False, port=5001, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        video_capture.stop()
