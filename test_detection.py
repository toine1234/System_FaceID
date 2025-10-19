from ultralytics import YOLO
from insightface.app import FaceAnalysis
from src.alignment import norm_crop
import cv2, numpy as np

class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="mps"):
        self.yolo = YOLO(yolo_model_path)
        self.device = device

        # RetinaFace hỗ trợ landmark 3D-68 (buffalo_l)
        self.face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if device != "cpu" else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        self.frame_count = 0
        self.retina_faces = []

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # --- Cập nhật khuôn mặt RetinaFace mỗi 5 frame ---
        if self.frame_count % 5 == 0:
            self.retina_faces = self.face_app.get(frame)
        retina_faces = getattr(self, "retina_faces", [])

        # --- YOLOv8n-Face 2D detection ---
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

                        # Landmark 2D
                        for (lx, ly) in kps:
                            cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

                        # Landmark 3D nếu có
                        if hasattr(face, "landmark_3d_68"):
                            lm3d = face.landmark_3d_68
                            for (x, y, z) in lm3d:
                                depth_color = int(np.clip((z - np.min(lm3d[:,2])) /
                                                          (np.ptp(lm3d[:,2]) + 1e-6) * 255, 0, 255))
                                cv2.circle(annotated, (int(x), int(y)), 1,
                                           (255 - depth_color, depth_color, 0), -1)

                        # Pose 3D (nếu model hỗ trợ)
                        if hasattr(face, "pose"):
                            yaw, pitch, roll = face.pose
                            cv2.putText(annotated,
                                        f"Y:{yaw:.1f} P:{pitch:.1f}",
                                        (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

                        aligned_faces.append((aligned, (x1, y1, x2, y2)))
                        found = True
                        break

                if not found:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        return annotated, aligned_faces

if __name__ == "__main__":
    print("🚀 Testing FaceDetector (YOLOv8n + RetinaFace 3D)...")

    detector = FaceDetector(device="mps")  # hoặc "cpu" nếu không có GPU

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, aligned_faces = detector.detect_and_align(frame)
        cv2.imshow("3D Face Detection (YOLOv8n + RetinaFace)", annotated)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("🛑 Đã thoát.")
            break

    cap.release()
    cv2.destroyAllWindows()
