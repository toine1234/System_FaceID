import os, time, cv2, numpy as np, torch, warnings, logging
from ultralytics import YOLO
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.app.common import Face
from torchvision.ops import nms
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov11n-face.pt", device=None,
                 face_app: Optional[FaceAnalysis]=None, yolo_imgsz=256,
                 yolo_conf=0.5, yolo_stride=3, predict_iou=0.4,
                 agnostic_nms=True, custom_nms_iou=0.4):
        self.frame_count, self.start_time, self.smooth_fps = 0, time.time(), 0.0
        self._cached_aligned_faces, self._frame_cache_ttl = [], 0
        self.device = device
        self.yolo_imgsz, self.yolo_conf, self.yolo_stride = yolo_imgsz, yolo_conf, yolo_stride
        self.predict_iou, self.agnostic_nms, self.custom_nms_iou = predict_iou, agnostic_nms, custom_nms_iou
        self.yolo = YOLO(yolo_model_path)
        self.yolo.fuse()
        self.landmark_model = face_app

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        
        if (self.frame_count % self.yolo_stride) == 0:
            results = self.yolo.predict(frame, imgsz=self.yolo_imgsz, conf=self.yolo_conf,
                                       iou=self.predict_iou, agnostic_nms=self.agnostic_nms,
                                       verbose=False, half=(self.device=="cuda"), device=self.device)
            boxes_obj = results[0].boxes if results and results[0] and results[0].boxes else None
            
            if not boxes_obj or len(boxes_obj) == 0:
                self.draw_fps(annotated)
                self._cached_aligned_faces = []
                return annotated, []
            
            boxes_xyxy, confs = boxes_obj.xyxy.cpu(), boxes_obj.conf.cpu()
            if len(boxes_xyxy) > 1:
                keep = nms(boxes_xyxy, confs, iou_threshold=self.custom_nms_iou)
                boxes_xyxy, confs = boxes_xyxy[keep], confs[keep]
            
            aligned_faces = []
            for (x1, y1, x2, y2), conf in zip(boxes_xyxy.numpy().astype(int), confs.numpy()):
                try:
                    face = Face(bbox=np.array([x1, y1, x2, y2]), det_score=conf)
                    lmk3d = self.landmark_model.get(frame, face)
                    kps = np.array([lmk3d[30], lmk3d[36], lmk3d[45], lmk3d[48], lmk3d[54]], dtype=np.float32)[:, :2]
                    aligned = face_align.norm_crop_with_landmark(frame, kps, 112)
                except:
                    y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
                    x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
                    aligned = cv2.resize(frame[y1_c:y2_c, x1_c:x2_c], (112, 112)) if (y2_c > y1_c and x2_c > x1_c) else np.zeros((112, 112, 3), dtype=np.uint8)
                
                if aligned is not None:
                    aligned_faces.append((aligned, (x1, y1, x2, y2)))
            
            self._cached_aligned_faces = aligned_faces
            self._frame_cache_ttl = self.yolo_stride * 2
        
        self._frame_cache_ttl -= 1
        if self._frame_cache_ttl <= 0:
            self._cached_aligned_faces = []
        
        self.draw_fps(annotated)
        return annotated, self._cached_aligned_faces

    def draw_fps(self, canvas):
        if self.frame_count % 15 == 0:
            elapsed = max(1e-6, time.time() - self.start_time)
            fps = self.frame_count / elapsed
            self.smooth_fps = fps if self.smooth_fps == 0 else (0.85 * self.smooth_fps + 0.15 * fps)
            if self.frame_count > 1000:
                self.frame_count, self.start_time = 0, time.time()
        
        cv2.putText(canvas, f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup",
                   (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)