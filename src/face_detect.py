import os
import time
import cv2
import numpy as np
import warnings
import torch
from ultralytics import YOLO
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import logging

# ================================================================
# ⚙️ Cấu hình log & cảnh báo
# ================================================================
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
ort.set_default_logger_severity(3)
logging.basicConfig(level=logging.ERROR)

class FaceDetector:
    def __init__(
        self,
        yolo_model_path="models/yolov11n-face.pt",
        device=None,
        yolo_imgsz=320,              # ✅ Giảm kích thước YOLO (từ 448 -> 320)
        yolo_conf=0.45,
        retina_det_size=(320, 320),  # ✅ Giảm kích thước RetinaFace
        retina_skip=12,              # ✅ Landmark mỗi 12 frame (thay vì 8)
        retina_conf=0.7,
        yolo_stride=2                # ✅ YOLO chạy mỗi 2 frame
    ):
        # --------- Thiết bị ----------
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"[INIT] FaceDetector initialized on {self.device}")

        # --------- YOLOv11 ----------
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy model YOLO: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        try:
            self.yolo.fuse()
        except Exception:
            pass
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        print(f"[INFO] YOLOv11 loaded: {os.path.basename(yolo_model_path)}")

        # --------- RetinaFace + 3D landmarks ----------
        ctx_id = 0 if self.device != "cpu" else -1
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_3d_68"]
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=retina_det_size)
        print(f"[INFO] RetinaFace 3D landmarks ready (det_size={retina_det_size})")

        # --------- Warm-up model (tránh giật khung đầu tiên) ----------
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = self.face_app.get(dummy)

        # --------- Trạng thái / cache ----------
        self.frame_count = 0
        self.start_time = time.time()
        self.smooth_fps = 0.0
        self.retina_cache = []
        self.retina_skip = retina_skip
        self.retina_conf = retina_conf
        self.yolo_stride = max(1, yolo_stride)
        self._yolo_boxes_cache = None

    # ----------------- Utilities -----------------
    @staticmethod
    def _iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (boxA[2]-boxA[0])) * max(1, (boxA[3]-boxA[1]))
        areaB = max(1, (boxB[2]-boxB[0])) * max(1, (boxB[3]-boxB[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    @staticmethod
    def _draw_landmarks_3d(canvas, lmk3d):
        lmk = lmk3d.astype(int)
        for i, (x, y, _) in enumerate(lmk):
            if 36 <= i <= 41:
                color = (0, 255, 0)
            elif 42 <= i <= 47:
                color = (255, 0, 0)
            elif 48 <= i <= 67:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)
            cv2.circle(canvas, (x, y), 1, color, -1)

    # ================================================================
    # 🚀 Phát hiện & Căn chỉnh khuôn mặt
    # ================================================================
    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # 1️⃣ RetinaFace mỗi retina_skip frame
        if self.frame_count % self.retina_skip == 0:
            faces = self.face_app.get(frame)
            self.retina_cache = [f for f in faces if getattr(f, "det_score", 0) >= self.retina_conf]
        retina_faces = self.retina_cache

        # 2️⃣ YOLO mỗi yolo_stride frame (còn lại dùng cache)
        run_yolo = (self.frame_count % self.yolo_stride == 0) or (self._yolo_boxes_cache is None)
        if run_yolo:
            results = self.yolo.predict(
                frame, imgsz=self.yolo_imgsz, conf=self.yolo_conf,
                verbose=False, half=(self.device == "cuda")
            )
            boxes = results[0].boxes if results and results[0] is not None else None
            self._yolo_boxes_cache = None if (boxes is None or len(boxes) == 0) else boxes.xyxy.cpu().numpy().astype(int)

        xyxy = self._yolo_boxes_cache
        if xyxy is None or len(xyxy) == 0:
            return annotated, aligned_faces

        # 3️⃣ Ghép bbox YOLO với RetinaFace (dựa IoU)
        for (x1, y1, x2, y2) in xyxy:
            box_yolo = [x1, y1, x2, y2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            best_face, best_iou = None, 0.0
            for f in retina_faces:
                bx1, by1, bx2, by2 = f.bbox.astype(int)
                iou = self._iou(box_yolo, [bx1, by1, bx2, by2])
                if iou > best_iou:
                    best_face, best_iou = f, iou

            aligned = None
            if best_face is not None and best_iou > 0.30:
                f = best_face
                if hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None:
                    lmk3d = f.landmark_3d_68
                    kps = np.array([lmk3d[30], lmk3d[36], lmk3d[45], lmk3d[48], lmk3d[54]], dtype=np.float32)[:, :2]
                    try:
                        aligned = face_align.norm_crop_with_landmark(frame, landmark=kps, image_size=112)
                    except Exception:
                        aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
                    # ✅ Chỉ vẽ landmark mỗi 5 frame
                    if self.frame_count % 5 == 0:
                        self._draw_landmarks_3d(annotated, lmk3d)
                elif hasattr(f, "kps") and f.kps is not None:
                    kps = f.kps.astype(np.float32)
                    aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
                    for (lx, ly) in kps.astype(int):
                        cv2.circle(annotated, (lx, ly), 2, (0, 0, 255), -1)
                else:
                    face_crop = frame[y1:y2, x1:x2]
                    aligned = cv2.resize(face_crop, (112, 112))
            else:
                face_crop = frame[y1:y2, x1:x2]
                aligned = cv2.resize(face_crop, (112, 112))

            aligned_faces.append((aligned, (x1, y1, x2, y2)))

        # 4️⃣ FPS tính mượt (EMA)
        if self.frame_count % 15 == 0:
            elapsed = time.time() - self.start_time
            fps_now = self.frame_count / max(1e-6, elapsed)
            self.smooth_fps = fps_now if self.smooth_fps == 0 else (0.85 * self.smooth_fps + 0.15 * fps_now)
            logging.info(f"[FPS] {self.smooth_fps:.1f}")

        # 5️⃣ Hiển thị FPS lên khung hình
        fps_text = f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup"
        cv2.putText(annotated, fps_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
        return annotated, aligned_faces
