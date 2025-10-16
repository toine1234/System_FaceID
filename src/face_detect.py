# Face Detection + Face Alignment

from ultralytics import YOLO
import cv2
import numpy as np
from src.alignment import norm_crop
class FaceDetector:
    def __init__(self, 
                 yolo_model_path="models/face_detector/yolov8n-face.pt"):
        self.model = YOLO(yolo_model_path)

    def detect_and_align(self, frame):
        """
        Phát hiện + căn chỉnh khuôn mặt dùng YOLOv8n-Face + ArcFace alignment
        Trả về frame có bounding boxes và danh sách khuôn mặt đã căn chỉnh
        """
        annotated_frame = frame.copy()
        aligned_faces = []

        results = self.model(frame, stream=True)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4]) if len(box) > 4 else 1.0

                # Vẽ bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Unknown", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if keypoints is not None and i < len(keypoints):
                    lmk = keypoints[i].astype(np.float32)
                    if lmk.shape == (5, 2):
                        aligned = norm_crop(frame, lmk, image_size=112, mode="arcface")
                        aligned_faces.append(aligned)

        return annotated_frame, aligned_faces
