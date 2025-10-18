from ultralytics import YOLO
from insightface.app import FaceAnalysis
from src.alignment import norm_crop
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, 
                 yolo_model_path="models/yolov8n-face.pt", 
                 device="cpu"):
        """
        YOLOv8n-face: phát hiện khuôn mặt.
        RetinaFace (InsightFace): lấy landmark 5 điểm chính xác.
        norm_crop(): căn chỉnh khuôn mặt theo chuẩn ArcFace.
        """
        self.yolo = YOLO(yolo_model_path)
        self.device = device
        self.frame_count = 0

        # --- Khởi tạo RetinaFace (InsightFace) ---
        self.face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if device in ["cuda", "mps"] else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print(f"[INFO] RetinaFace đã sẵn sàng ({device})")

    def detect_and_align(self, frame):
        """
        Phát hiện bằng YOLO, căn chỉnh bằng RetinaFace.
        Trả về:
            - annotated: frame có vẽ khung & landmark
            - aligned_faces: [(face_img, (x1, y1, x2, y2))]
        """
        annotated = frame.copy()
        aligned_faces = []

        # --- YOLO phát hiện khuôn mặt ---
        results = self.yolo(frame, imgsz=384, conf=0.5, iou=0.45, verbose=False, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                w, h = x2 - x1, y2 - y1
                if w < 50 or h < 50:
                    continue

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    # --- Lấy landmark bằng RetinaFace ---
                    faces = self.face_app.get(face_crop)
                    if len(faces) > 0:
                        # Chọn khuôn mặt lớn nhất
                        main_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                        landmark = main_face.landmark.astype(np.float32)

                        # Điều chỉnh toạ độ landmark về toàn frame
                        landmark[:, 0] += x1
                        landmark[:, 1] += y1

                        # --- Căn chỉnh khuôn mặt ---
                        aligned = norm_crop(frame, landmark, image_size=112)

                        # --- Vẽ bounding box & landmark ---
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        for (lx, ly) in landmark:
                            cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

                        aligned_faces.append((aligned, (x1, y1, x2, y2)))
                    else:
                        print("[WARN] ❌ Không tìm thấy landmark, dùng crop gốc.")
                        aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

                except Exception as e:
                    print(f"[WARN] ⚠️ Lỗi alignment RetinaFace: {e}")
                    aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        return annotated, aligned_faces
