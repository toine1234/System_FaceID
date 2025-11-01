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
from insightface.app.common import Face
from typing import Optional, List, Tuple

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
        face_app: Optional[FaceAnalysis] = None,
        yolo_imgsz=320,
        yolo_conf=0.45,
        yolo_stride=3,
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
        print(f"[INFO] YOLOv11 loaded (stride={self.yolo_stride})")

        # --------- InsightFace (LANDMARKS) ----------
        if face_app:
            self.face_app = face_app
            print("[INFO] InsightFace Landmark (Shared) ready")
        else:
            print("[WARN] Creating new FaceAnalysis for Detector (Landmarks only).")
            ctx_id = 0 if self.device != "cpu" else -1
            self.face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["landmark_3d_68"])
            self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        if 'landmark_3d_68' not in self.face_app.models:
             raise RuntimeError("FaceAnalysis object must be initialized with 'landmark_3d_68'.")
        
        self.landmark_model = self.face_app.models['landmark_3d_68']

        # --------- States & Cache----------
        self.frame_count, self.start_time, self.smooth_fps = 0, time.time(), 0.0
        
        # ĐẢM BẢO BẠN CÓ DÒNG NÀY (CACHE MỚI)
        self._cached_aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []    
    
    # ================================================================
    #  Utilities
    # ================================================================
    
    # _iou no longer needed

    @staticmethod
    def _draw_landmarks(canvas, lmk3d):
        """Vẽ 68 điểm mốc 3D lên canvas."""
        for i, (x, y, _) in enumerate(lmk3d.astype(int)):
            color = (
                (0, 255, 0) if 36 <= i <= 41 else  # Mắt
                (255, 0, 0) if 42 <= i <= 47 else  # Mắt
                (0, 255, 255) if 48 <= i <= 67 else # Môi
                (200, 200, 200) # Khác
            )
            cv2.circle(canvas, (x, y), 1, color, -1)

    # ================================================================
    # Detection & Alignment (LOGIC MỚI)
    # ================================================================
    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy() # Luôn bắt đầu từ frame mới
        
        # DÒNG IF NÀY CHỈ CHECK 1 ĐIỀU KIỆN (đây là code đúng)
        if self.frame_count % self.yolo_stride != 0:
            
            # 1. Reset cache
            aligned_faces_results = []

            # 2. CHẠY YOLO DETECTION
            results = self.yolo.predict(frame, imgsz=self.yolo_imgsz, conf=self.yolo_conf,
                                        verbose=False, half=(self.device == "cuda"))
            
            boxes_obj = None
            if results and results[0] and results[0].boxes:
                boxes_obj = results[0].boxes

            if boxes_obj is None or len(boxes_obj) == 0:
                self.draw_fps(annotated)
                self._cached_aligned_faces = [] # Cache rỗng
                return annotated, []

            boxes_xyxy = boxes_obj.xyxy.cpu().numpy().astype(int)
            confs = boxes_obj.conf.cpu().numpy()

            # 3. CHẠY LANDMARK & ALIGNMENT CHO TỪNG BOX
            for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                
                aligned = None
                bbox = (x1, y1, x2, y2)
                try:
                    face = Face(bbox=np.array([x1, y1, x2, y2]), det_score=conf)
                    lmk3d = self.landmark_model.get(frame, face)
                    kps = np.array([lmk3d[30], lmk3d[36], lmk3d[45], lmk3d[48], lmk3d[54]], dtype=np.float32)[:, :2]
                    aligned = face_align.norm_crop_with_landmark(frame, kps, 112)
                        
                except Exception as e:
                    y1_c, y2_c = max(0, y1), max(0, y2)
                    x1_c, x2_c = max(0, x1), max(0, x2)
                    if y2_c > y1_c and x2_c > x1_c:
                         aligned = cv2.resize(frame[y1_c:y2_c, x1_c:x2_c], (112, 112))
                    else:
                        aligned = np.zeros((112, 112, 3), dtype=np.uint8)

                if aligned is not None:
                    aligned_faces_results.append((aligned, bbox))
            
            self.draw_fps(annotated) # Vẽ FPS
            
            # 4. LƯU VÀO CACHE
            self._cached_aligned_faces = aligned_faces_results
            
            return annotated, aligned_faces_results
        
        # TRẢ VỀ KẾT QUẢ TỪ CACHE (CHO CÁC FRAME ĐỆM)
        else:
            self.draw_fps(annotated) # Vẫn vẽ FPS
            return annotated, self._cached_aligned_faces

    def draw_fps(self, canvas):
        """Tính toán và vẽ FPS (làm mượt EMA) lên canvas."""
        if self.frame_count % 15 == 0: # Cập nhật FPS sau mỗi 15 frames
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.frame_count / elapsed
            # Dùng Exponential Moving Average để làm mượt FPS
            self.smooth_fps = fps if self.smooth_fps == 0 else (0.85 * self.smooth_fps + 0.15 * fps)
            
            # Reset bộ đếm để tránh tràn số và giữ FPS ổn định
            if self.frame_count > 1000:
                self.frame_count = 0
                self.start_time = time.time()

        cv2.putText(canvas, f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)