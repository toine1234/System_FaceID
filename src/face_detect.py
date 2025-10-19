# from ultralytics import YOLO
# from facenet_pytorch import MTCNN
# from src.alignment import norm_crop
# import cv2
# import numpy as np


# class FaceDetector:
#     def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="cpu"):
#         self.yolo = YOLO(yolo_model_path)
#         self.mtcnn = MTCNN(keep_all=True, device=device)
#         self.device = device
#         self.frame_count = 0


#     def detect_and_align(self, frame):
#         self.frame_count += 1
#         annotated = frame.copy()
#         aligned_faces = []

#         # --- Phát hiện khuôn mặt bằng YOLO ---
#         results = self.yolo(frame, imgsz=384, conf=0.5, iou=0.45, verbose=False, stream=True)
#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue

#                 # --- MTCNN Landmark ---
#                 try:
#                     _, _, points = self.mtcnn.detect(face_crop, landmarks=True)
#                     if points is not None and len(points) > 0:
#                         landmark = np.array(points[0], dtype=np.float32)

#                         # --- Căn chỉnh bằng ArcFace norm_crop ---
#                         aligned = norm_crop(face_crop, landmark, image_size=112)

#                         # Vẽ bounding box & landmark lên frame
#                         cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         for (lx, ly) in landmark:
#                             cv2.circle(annotated, (int(x1 + lx), int(y1 + ly)), 2, (0, 0, 255), -1)

#                         aligned_faces.append((aligned, (x1, y1, x2, y2)))

#                     else:
#                         print("[WARN] Không tìm thấy landmark, dùng crop gốc.")
#                         aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

#                 except Exception as e:
#                     print(f"[WARN] lỗi alignment: {e}")
#                     aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

#         return annotated, aligned_faces


from ultralytics import YOLO
from insightface.app import FaceAnalysis
from src.alignment import norm_crop
import cv2, numpy as np

class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="mps"):
        self.yolo = YOLO(yolo_model_path)
        self.device = device

        # RetinaFace nhẹ hơn (320x320)
        self.face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if device != "cpu" else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        self.frame_count = 0
        self.retina_faces = []

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # --- Chạy RetinaFace mỗi 5 frame ---
        if self.frame_count % 5 == 0:
            self.retina_faces = self.face_app.get(frame)

        retina_faces = getattr(self, "retina_faces", [])

        # --- YOLOv8n-Face ---
        results = self.yolo(frame, imgsz=256, conf=0.5, iou=0.45, verbose=False, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                found = False
                for face in retina_faces:
                    bx1, by1, bx2, by2 = face.bbox.astype(int)
                    if (bx1 >= x1 - 15 and by1 >= y1 - 15 and bx2 <= x2 + 15 and by2 <= y2 + 15):
                        kps = face.kps.astype(np.float32)
                        aligned = norm_crop(frame, kps, image_size=112)

                        # vẽ landmark (tùy chọn)
                        for (lx, ly) in kps:
                            cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

                        aligned_faces.append((aligned, (x1, y1, x2, y2)))
                        found = True
                        break

                if not found:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        return annotated, aligned_faces


# if __name__ == "__main__":
#     detector = FaceDetector(device="mps")  # hoặc "cpu"
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated, aligned_faces = detector.detect_and_align(frame)
#         cv2.imshow("Optimized YOLO + RetinaFace", annotated)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

