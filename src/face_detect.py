from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path="models/face_detector/yolov8n-face/train_results/weights/best.pt"):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model(frame, stream = True)
        for r in results:
            boxes = r.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame