import os
import time
import cv2
import numpy as np
import torch
import warnings
import logging
from ultralytics import YOLO
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align


# ================================================================
# Logging & Warning Setup
# ================================================================
warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


class FaceDetector:
    def __init__(
        self,
        yolo_model_path="models/yolov11n-face.pt",
        device=None,
        yolo_imgsz=320,
        yolo_conf=0.45,
        retina_det_size=(320, 320),
        retina_skip=12,
        retina_conf=0.7,
        yolo_stride=2,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[INIT] FaceDetector on {self.device}")

        # --------- YOLOv11 ----------
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"❌ Missing YOLO model: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        self.yolo_imgsz, self.yolo_conf, self.yolo_stride = yolo_imgsz, yolo_conf, max(1, yolo_stride)
        try:
            self.yolo.fuse()
        except Exception:
            pass
        print(f"[INFO] YOLOv11 loaded ({os.path.basename(yolo_model_path)})")

        # --------- RetinaFace ----------
        ctx_id = 0 if self.device != "cpu" else -1
        self.face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"])
        self.face_app.prepare(ctx_id=ctx_id, det_size=retina_det_size)
        print(f"[INFO] RetinaFace 3D ready (det_size={retina_det_size})")

        # Warm-up
        self.face_app.get(np.zeros((320, 320, 3), dtype=np.uint8))

        # --------- States ----------
        self.frame_count, self.start_time, self.smooth_fps = 0, time.time(), 0.0
        self.retina_cache, self._yolo_boxes_cache = [], None
        self.retina_skip, self.retina_conf = retina_skip, retina_conf

    # ================================================================
    #  Utilities
    # ================================================================
    @staticmethod
    def _iou(a, b):
        xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (a[2] - a[0])) * max(1, (a[3] - a[1]))
        areaB = max(1, (b[2] - b[0])) * max(1, (b[3] - b[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    @staticmethod
    def _draw_landmarks(canvas, lmk3d):
        for i, (x, y, _) in enumerate(lmk3d.astype(int)):
            color = (
                (0, 255, 0) if 36 <= i <= 41 else
                (255, 0, 0) if 42 <= i <= 47 else
                (0, 255, 255) if 48 <= i <= 67 else
                (200, 200, 200)
            )
            cv2.circle(canvas, (x, y), 1, color, -1)

    # ================================================================
    # Detection & Alignment
    # ================================================================
    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated, aligned_faces = frame.copy(), []

        # RetinaFace cache update
        if self.frame_count % self.retina_skip == 0:
            faces = self.face_app.get(frame)
            self.retina_cache = [f for f in faces if getattr(f, "det_score", 0) >= self.retina_conf]

        # YOLO update
        if self.frame_count % self.yolo_stride == 0 or self._yolo_boxes_cache is None:
            results = self.yolo.predict(frame, imgsz=self.yolo_imgsz, conf=self.yolo_conf,
                                        verbose=False, half=(self.device == "cuda"))
            boxes = results[0].boxes if results and results[0] is not None else None
            self._yolo_boxes_cache = None if not boxes else boxes.xyxy.cpu().numpy().astype(int)

        boxes = self._yolo_boxes_cache
        if boxes is None or len(boxes) == 0:
            return annotated, aligned_faces

        # Match YOLO ↔ RetinaFace
        for (x1, y1, x2, y2) in boxes:
            box_yolo = [x1, y1, x2, y2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Best Retina match
            best_face, best_iou = max(
                ((f, self._iou(box_yolo, f.bbox.astype(int))) for f in self.retina_cache),
                key=lambda x: x[1], default=(None, 0)
            )

            aligned = None
            if best_face and best_iou > 0.3:
                f = best_face
                if hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None:
                    lmk3d = f.landmark_3d_68
                    kps = np.array([lmk3d[30], lmk3d[36], lmk3d[45], lmk3d[48], lmk3d[54]], dtype=np.float32)[:, :2]
                    try:
                        aligned = face_align.norm_crop_with_landmark(frame, kps, 112)
                    except Exception:
                        aligned = face_align.norm_crop(frame, kps, 112)
                    if self.frame_count % 5 == 0:
                        self._draw_landmarks(annotated, lmk3d)
                elif hasattr(f, "kps") and f.kps is not None:
                    kps = f.kps.astype(np.float32)
                    aligned = face_align.norm_crop(frame, kps, 112)
                    for (lx, ly) in kps.astype(int):
                        cv2.circle(annotated, (lx, ly), 2, (0, 0, 255), -1)
                else:
                    aligned = cv2.resize(frame[y1:y2, x1:x2], (112, 112))
            else:
                aligned = cv2.resize(frame[y1:y2, x1:x2], (112, 112))

            aligned_faces.append((aligned, (x1, y1, x2, y2)))

        # FPS (EMA)
        if self.frame_count % 15 == 0:
            fps = self.frame_count / max(1e-6, time.time() - self.start_time)
            self.smooth_fps = fps if self.smooth_fps == 0 else 0.85 * self.smooth_fps + 0.15 * fps
            logging.info(f"[FPS] {self.smooth_fps:.1f}")

        cv2.putText(annotated, f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
        return annotated, aligned_faces