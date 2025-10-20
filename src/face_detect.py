import os
import time
import cv2
import numpy as np
import warnings
import torch
from concurrent.futures import ThreadPoolExecutor, Future
from ultralytics import YOLO
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceDetector:
    def __init__(
        self,
        yolo_model_path="models/yolov11n-face.pt",
        device=None,
        yolo_imgsz=448,
        yolo_conf=0.45,
        retina_det_size=(640, 640),
        retina_skip=8,
        retina_conf=0.7,
    ):
        # Ẩn warning không cần thiết
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
        warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
        ort.set_default_logger_severity(3)  # Ẩn log onnxruntime

        # --------- Thiết bị ----------
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"[INIT] FaceDetector initialized on {self.device}")

        # --------- YOLOv11 ----------
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy model YOLO: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        try:
            self.yolo.fuse()
        except Exception:
            pass
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        print(f"[INFO] YOLOv11 loaded: {os.path.basename(yolo_model_path)}")

        # --------- RetinaFace + 3D landmarks ----------
        ctx_id = 0 if self.device != "cpu" else -1
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_3d_68"]
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=retina_det_size)
        print(f"[INFO] RetinaFace 3D landmarks ready (det_size={retina_det_size})")

        # --------- Trạng thái / cache ----------
        self.frame_count = 0
        self.start_time = time.time()
        self.smooth_fps = 0.0

        self.retina_cache = []
        self.retina_skip = retina_skip
        self.retina_conf = retina_conf

    # ----------------- Utilities -----------------
    @staticmethod
    def _iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (boxA[2]-boxA[0])) * max(1, (boxA[3]-boxA[1]))
        areaB = max(1, (boxB[2]-boxB[0])) * max(1, (boxB[3]-boxB[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    @staticmethod
    def _draw_landmarks_3d(canvas, lmk3d):
        lmk = lmk3d.astype(int)
        for i, (x, y, _) in enumerate(lmk):
            if 36 <= i <= 41:
                color = (0, 255, 0)
            elif 42 <= i <= 47:
                color = (255, 0, 0)
            elif 48 <= i <= 67:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)
            cv2.circle(canvas, (x, y), 1, color, -1)

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        if self.frame_count % self.retina_skip == 0:
            faces = self.face_app.get(frame)
            self.retina_cache = [f for f in faces if getattr(f, "det_score", 0) >= self.retina_conf]
        retina_faces = self.retina_cache

        # YOLO: phát hiện bbox nhanh
        results = self.yolo.predict(
            frame, imgsz=self.yolo_imgsz, conf=self.yolo_conf,
            verbose=False, half=(self.device == "cuda")
        )
        boxes = results[0].boxes if results and results[0] is not None else None
        if boxes is None or len(boxes) == 0:
            return annotated, aligned_faces

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in xyxy:
            box_yolo = [x1, y1, x2, y2]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ghép YOLO với RetinaFace theo IoU cao nhất
            best_face, best_iou = None, 0.0
            for f in retina_faces:
                bx1, by1, bx2, by2 = f.bbox.astype(int)
                iou = self._iou(box_yolo, [bx1, by1, bx2, by2])
                if iou > best_iou:
                    best_face, best_iou = f, iou

            aligned = None
            if best_face is not None and best_iou > 0.30:
                f = best_face
                # Ưu tiên landmark 3D 68 điểm
                if hasattr(f, "landmark_3d_68") and f.landmark_3d_68 is not None:
                    lmk3d = f.landmark_3d_68
                    # 5 điểm chủ chốt theo ArcFace (mũi 30, mắt 36/45, mép 48/54)
                    kps = np.array([lmk3d[30], lmk3d[36], lmk3d[45], lmk3d[48], lmk3d[54]], dtype=np.float32)[:, :2]
                    try:
                        aligned = face_align.norm_crop_with_landmark(frame, landmark=kps, image_size=112)
                    except Exception:
                        aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)

                    self._draw_landmarks_3d(annotated, lmk3d)

                # Fallback: kps 5 điểm 2D
                elif hasattr(f, "kps") and f.kps is not None:
                    kps = f.kps.astype(np.float32)  # (5,2)
                    aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
                    for (lx, ly) in kps.astype(int):
                        cv2.circle(annotated, (lx, ly), 2, (0, 0, 255), -1)

                # Fallback cuối: crop bbox YOLO
                else:
                    face_crop = frame[y1:y2, x1:x2]
                    aligned = cv2.resize(face_crop, (112, 112))

            else:
                face_crop = frame[y1:y2, x1:x2]
                aligned = cv2.resize(face_crop, (112, 112))

            aligned_faces.append((aligned, (x1, y1, x2, y2)))

        # Tính FPS mượt (EMA)
        if self.frame_count % 15 == 0:  # giảm tần suất log FPS
            elapsed = time.time() - self.start_time
            fps_now = self.frame_count / max(1e-6, elapsed)
            self.smooth_fps = fps_now if self.smooth_fps == 0 else (0.85 * self.smooth_fps + 0.15 * fps_now)
            print(f"[FPS] {self.smooth_fps:.1f}")

        # Vẽ FPS lên khung hình
        fps_text = f"FPS: {self.smooth_fps:.1f}" if self.smooth_fps else "FPS: warmup"
        cv2.putText(annotated, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2, cv2.LINE_AA)

        return annotated, aligned_faces

# def run_realtime():
#     print("[TEST] Running FaceDetector – YOLOv11 + RetinaFace 3D (Async/MPS)")

#     detector = FaceDetector(
#         yolo_model_path="/Users/sarahtruc/Documents/System_FaceID/models/yolov11n-face.pt",
#         device="mps",
#         yolo_imgsz=448,
#         yolo_conf=0.45,
#         retina_det_size=(320, 320),
#         retina_skip=8,
#         retina_conf=0.70,
#     )

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     if not cap.isOpened():
#         print("❌ Không thể mở webcam!")
#         return

#     print("[INFO] Nhấn 'q' để thoát…")

#     # --- Double-buffer bất đồng bộ ---
#     executor = ThreadPoolExecutor(max_workers=2)

#     # Lấy khung đầu tiên
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Không lấy được khung hình đầu tiên")
#         cap.release()
#         return

#     future: Future = executor.submit(detector.detect_and_align, frame.copy())

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Lấy kết quả đã xử lý của khung trước
#         annotated_prev, _ = future.result()

#         # Gửi khung hiện tại đi xử lý
#         future = executor.submit(detector.detect_and_align, frame.copy())

#         # Hiển thị khung đã xử lý
#         cv2.imshow("YOLOv11 + RetinaFace 3D (Optimized Async)", annotated_prev)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     executor.shutdown(wait=False)
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run_realtime()