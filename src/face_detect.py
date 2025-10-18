from ultralytics import YOLO
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from src.alignment import norm_crop


class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="cpu"):
        """
        YOLOv8n-face chỉ dùng để phát hiện khuôn mặt.
        MTCNN dùng để căn chỉnh khuôn mặt bằng landmark (5 điểm).
        """
        self.yolo = YOLO(yolo_model_path)
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.device = device
        self.frame_count = 0


    def detect_and_align(self, frame):
        """
        Phát hiện khuôn mặt bằng YOLO + căn chỉnh bằng MTCNN.
        Trả về:
            - annotated: frame có vẽ khung + landmark
            - aligned_faces: [(face_img, (x1, y1, x2, y2))]
        """
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # 1️⃣ Phát hiện khuôn mặt
        results = self.yolo(frame, imgsz=384, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                # 2️⃣ Căn chỉnh chỉ mỗi 3 frame
                if self.frame_count % 3 == 0:
                    try:
                        _, _, points = self.mtcnn.detect(face_crop, landmarks=True)
                        if points is not None:
                            landmark = np.array(points[0], dtype=np.float32)
                            aligned = norm_crop(face_crop, landmark)
                            for (lx, ly) in landmark:
                                cv2.circle(annotated, (int(x1 + lx), int(y1 + ly)), 2, (0, 0, 255), -1)
                        else:
                            aligned = face_crop
                    except Exception as e:
                        print(f"[WARN] alignment error: {e}")
                        aligned = face_crop
                else:
                    aligned = face_crop

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                aligned_faces.append((aligned, (x1, y1, x2, y2)))

        return annotated, aligned_faces