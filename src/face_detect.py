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
from torchvision.ops import nms  # THÊM: Cho custom NMS
from typing import Optional, List, Tuple

# ================================================================
# Logging & Warning Setup (KHÔNG THAY ĐỔI)
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
        yolo_imgsz=256,  # GIẢM: Input size nhỏ hơn (từ 320)
        yolo_conf=0.5,
        yolo_stride=3,  # TĂNG: Internal stride (từ 2)
        predict_iou=0.4,  # TĂNG: NMS IoU cao hơn cho nhanh (từ 0.3)
        agnostic_nms=True,
        custom_nms_iou=0.4,  # Đồng bộ
    ):

        self.frame_count, self.start_time, self.smooth_fps = 0, time.time(), 0.0
        self._last_detection_frame = -999
        self._cached_aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        self._frame_cache_ttl = 0  # TĂNG TTL= yolo_stride * 1.5 nếu cần
        self.device = device
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.yolo_stride = yolo_stride
        self.predict_iou = predict_iou
        self.agnostic_nms = agnostic_nms
        self.custom_nms_iou = custom_nms_iou
        
        # OPTIMIZED: Load YOLO với half và engine nếu có
        self.yolo = YOLO(yolo_model_path)
        half = (self.device == "cuda")  # Bật FP16 nếu CUDA
        self.yolo.fuse()  # Fuse layers cho speed
        self.landmark_model = face_app

    # ================================================================
    # Utility Functions (KHÔNG THAY ĐỔI)
    # ================================================================
    @staticmethod
    def _draw_landmarks(canvas, lmk3d):
        """Vẽ 68 điểm mốc 3D lên canvas."""
        for i, (x, y, _) in enumerate(lmk3d.astype(int)):
            color = (
                (0, 255, 0) if 36 <= i <= 41 else  # Mắt trái
                (255, 0, 0) if 42 <= i <= 47 else  # Mắt phải
                (0, 255, 255) if 48 <= i <= 67 else  # Môi
                (200, 200, 200)  # Khác
            )
            cv2.circle(canvas, (x, y), 1, color, -1)

    # ================================================================
    # Detection & Alignment OPTIMIZED
    # ================================================================
    def detect_and_align(self, frame):
        """Phát hiện khuôn mặt và căn chỉnh theo landmark."""
        self.frame_count += 1
        annotated = frame.copy()

        should_detect = (self.frame_count % self.yolo_stride) == 0
        
        if should_detect:
            # 1️⃣ YOLO Detection OPTIMIZED: half, verbose=False, engine auto
            results = self.yolo.predict(
                frame,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                iou=self.predict_iou,
                agnostic_nms=self.agnostic_nms,
                verbose=False,
                half=(self.device == "cuda"),  # FP16 cho CUDA
                device=self.device  # Explicit device
            )

            boxes_obj = None
            if results and results[0] and results[0].boxes:
                boxes_obj = results[0].boxes

            if boxes_obj is None or len(boxes_obj) == 0:
                self.draw_fps(annotated)
                self._cached_aligned_faces = []
                self._last_detection_frame = self.frame_count
                return annotated, []

            # 1️⃣.5️⃣ Custom NMS fallback
            boxes_xyxy = boxes_obj.xyxy.cpu()
            confs = boxes_obj.conf.cpu()

            if len(boxes_xyxy) > 1:
                keep_indices = nms(
                    boxes_xyxy,
                    confs,
                    iou_threshold=self.custom_nms_iou
                )
                boxes_xyxy = boxes_xyxy[keep_indices]
                confs = confs[keep_indices]

            boxes_xyxy = boxes_xyxy.numpy().astype(int)
            confs = confs.numpy()

            # 2️⃣ Landmark & Alignment (batch nếu multi-face)
            aligned_faces_results = []
            for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                aligned = None
                bbox = (x1, y1, x2, y2)
                try:
                    face = Face(bbox=np.array([x1, y1, x2, y2]), det_score=conf)
                    lmk3d = self.landmark_model.get(frame, face)
                    kps = np.array([
                        lmk3d[30],  # Mũi
                        lmk3d[36],  # Mắt trái
                        lmk3d[45],  # Mắt phải
                        lmk3d[48],  # Mép trái
                        lmk3d[54]   # Mép phải
                    ], dtype=np.float32)[:, :2]
                    aligned = face_align.norm_crop_with_landmark(frame, kps, 112)

                except Exception:
                    y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
                    x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
                    if y2_c > y1_c and x2_c > x1_c:
                        aligned = cv2.resize(frame[y1_c:y2_c, x1_c:x2_c], (112, 112))
                    else:
                        aligned = np.zeros((112, 112, 3), dtype=np.uint8)

                if aligned is not None:
                    aligned_faces_results.append((aligned, bbox))

            self._cached_aligned_faces = aligned_faces_results
            self._frame_cache_ttl = self.yolo_stride * 2  # TĂNG: Cache lâu hơn (từ stride)

        self._frame_cache_ttl -= 1
        if self._frame_cache_ttl <= 0:
            self._cached_aligned_faces = []

        self.draw_fps(annotated)
        return annotated, self._cached_aligned_faces

    # ================================================================
    # FPS Calculation (KHÔNG THAY ĐỔI)
    # ================================================================
    def draw_fps(self, canvas):
        """Tính toán và vẽ FPS (làm mượt EMA) lên canvas."""
        if self.frame_count % 15 == 0:
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.frame_count / elapsed
            self.smooth_fps = fps if self.smooth_fps == 0 else (0.85 * self.smooth_fps + 0.15 * fps)
            if self.frame_count > 1000:
                self.frame_count = 0
                self.start_time = time.time()

        cv2.putText(
            canvas,
            f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 80),
            2,
            cv2.LINE_AA
        )