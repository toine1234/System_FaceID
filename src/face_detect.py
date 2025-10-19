import cv2
import numpy as np
import time
import torch
import warnings
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class FaceDetector:
    def __init__(self, yolo_model_path="/Users/sarahtruc/Documents/System_FaceID/models/yolov11n-face.pt", device=None):
        #Thiết bị
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"[INFO] FaceDetector initialized on device: {self.device}")

        #YOLOv11 (detection)
        try:
            self.yolo = YOLO(yolo_model_path)
            print(f"[INFO] YOLOv11 model loaded successfully: {yolo_model_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Cannot load YOLO model: {e}")

        #RetinaFace (buffalo_l) — chỉ bật detection + landmark 3D
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_3d_68"]
        )
        # Giảm kích thước input để tăng FPS (480x480)
        self.face_app.prepare(ctx_id=0, det_size=(480, 480))
        print("[INFO] RetinaFace (buffalo_l) initialized with 3D landmarks (480×480)")

        #Biến trạng thái
        self.frame_count = 0
        self.retina_faces = []
        self.start_time = time.time()

    # ------------------------------------------------------------
    def _iou(self, boxA, boxB):
        """Tính IoU giữa 2 hộp"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    # ------------------------------------------------------------
    def detect_and_align(self, frame):
        """Phát hiện YOLO và căn chỉnh RetinaFace 3D"""
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        #RetinaFace mỗi 5 frame để giảm tải CPU
        if self.frame_count % 5 == 0:
            faces = self.face_app.get(frame)
            # Giữ những mặt confidence cao
            self.retina_faces = [f for f in faces if getattr(f, "det_score", 0) > 0.7]
        retina_faces = getattr(self, "retina_faces", [])

        # YOLOv11
        results = self.yolo(frame, verbose=False)
        if not results:
            return annotated, aligned_faces
        boxes = results[0].boxes
        if boxes is None or boxes.shape[0] == 0:
            return annotated, aligned_faces

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in xyxy:
            box_yolo = [x1, y1, x2, y2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Chọn mặt Retina trùng YOLO cao nhất theo IoU
            best_face, best_iou = None, 0
            for face in retina_faces:
                bx1, by1, bx2, by2 = face.bbox.astype(int)
                iou = self._iou(box_yolo, [bx1, by1, bx2, by2])
                if iou > best_iou:
                    best_face, best_iou = face, iou

            if best_face is not None and best_iou > 0.3:
                face = best_face

                # Landmark 3D ưu tiên
                if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                    lmk3d = face.landmark_3d_68
                    kps_3d = np.array([
                        lmk3d[30],  # mũi
                        lmk3d[36],  # mắt trái
                        lmk3d[45],  # mắt phải
                        lmk3d[48],  # mép trái
                        lmk3d[54],  # mép phải
                    ], dtype=np.float32)
                    kps_2d = kps_3d[:, :2]

                    # 🧠 Alignment 3D chính xác hơn
                    try:
                        aligned = face_align.norm_crop_with_landmark(frame, landmark=kps_2d, image_size=112)
                    except Exception:
                        aligned = face_align.norm_crop(frame, landmark=kps_2d, image_size=112)

                    # 🎨 Vẽ landmark 3D với màu vàng rõ nét
                    for i, (lx, ly, _) in enumerate(lmk3d.astype(int)):
                        if i in range(36, 42):       # mắt trái
                            color = (0, 255, 0)
                        elif i in range(42, 48):     # mắt phải
                            color = (255, 0, 0)
                        elif i in range(48, 68):     # miệng
                            color = (0, 255, 255)
                        else:
                            color = (200, 200, 200)
                        cv2.circle(annotated, (lx, ly), 2, color, -1)

                elif hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps.astype(np.float32)
                    aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
                    for (lx, ly) in kps.astype(int):
                        cv2.circle(annotated, (lx, ly), 3, (0, 0, 255), -1)
                else:
                    face_crop = frame[y1:y2, x1:x2]
                    aligned = cv2.resize(face_crop, (112, 112))

                aligned_faces.append((aligned, (x1, y1, x2, y2)))

            else:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        # Log FPS
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"[INFO] Processed {self.frame_count} frames ({fps:.2f} FPS)")

        return annotated, aligned_faces

if __name__ == "__main__":
    print("[TEST] Running optimized FaceDetector (YOLOv11 + RetinaFace 3D)...")

    detector = FaceDetector(
        yolo_model_path="/Users/sarahtruc/Documents/System_FaceID/models/yolov11n-face.pt",
        device="mps"
    )

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)   # độ phân giải đầu vào
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        exit()

    print("[INFO] Nhấn 'q' để thoát chương trình...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, aligned_faces = detector.detect_and_align(frame)
        cv2.imshow("YOLOv11 + RetinaFace 3D (Optimized)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()