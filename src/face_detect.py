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
        annotated = frame.copy()
        aligned_faces = []

        results = self.yolo(frame, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Cắt khuôn mặt từ ảnh gốc
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                aligned = face_crop  # mặc định nếu không alignment được

                try:
                    # MTCNN detect landmark (đã resize nội bộ)
                    _, _, points = self.mtcnn.detect(face_crop, landmarks=True)

                    if points is not None and len(points) > 0:
                        landmark = np.array(points[0], dtype=np.float32)

                        # Căn chỉnh theo ArcFace chuẩn 112x112
                        aligned = norm_crop(face_crop, landmark)

                        # Vẽ landmark lên ảnh hiển thị
                        for (lx, ly) in landmark:
                            cv2.circle(annotated, (int(x1 + lx), int(y1 + ly)), 2, (0, 0, 255), -1)

                    else:
                        print("[WARN] ❌ Không tìm thấy landmark, dùng crop gốc.")

                except Exception as e:
                    print(f"[WARN] ⚠️ Lỗi alignment MTCNN: {e}")

                aligned_faces.append((aligned, (x1, y1, x2, y2)))

        return annotated, aligned_faces