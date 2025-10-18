from ultralytics import YOLO
from facenet_pytorch import MTCNN
from src.alignment import norm_crop
import cv2
import numpy as np
from insightface.app import FaceAnalysis



class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="cpu"):
        self.yolo = YOLO(yolo_model_path)
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.device = device
        self.frame_count = 0


    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # --- Phát hiện khuôn mặt bằng YOLO ---
        results = self.yolo(frame, imgsz=384, conf=0.5, iou=0.45, verbose=False, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # --- MTCNN Landmark ---
                try:
                    _, _, points = self.mtcnn.detect(face_crop, landmarks=True)
                    if points is not None and len(points) > 0:
                        landmark = np.array(points[0], dtype=np.float32)

                        # --- Căn chỉnh bằng ArcFace norm_crop ---
                        aligned = norm_crop(face_crop, landmark, image_size=112)

                        # Vẽ bounding box & landmark lên frame
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        for (lx, ly) in landmark:
                            cv2.circle(annotated, (int(x1 + lx), int(y1 + ly)), 2, (0, 0, 255), -1)

                        aligned_faces.append((aligned, (x1, y1, x2, y2)))

                    else:
                        print("[WARN] Không tìm thấy landmark, dùng crop gốc.")
                        aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

                except Exception as e:
                    print(f"[WARN] lỗi alignment: {e}")
                    aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        return annotated, aligned_faces
