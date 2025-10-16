from ultralytics import YOLO
from mtcnn import MTCNN
import cv2
import numpy as np
from src.alignment import norm_crop

class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt"):
        self.yolo = YOLO(yolo_model_path)
        self.mtcnn = MTCNN()
        self.frame_count = 0

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        results = self.yolo(frame, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # chỉ chạy MTCNN mỗi 3 frame
                if self.frame_count % 3 != 0:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                detections = self.mtcnn.detect_faces(rgb_crop)
                if not detections:
                    continue

                keypoints = detections[0]['keypoints']
                lmk = np.array([
                    [keypoints['left_eye'][0] + x1, keypoints['left_eye'][1] + y1],
                    [keypoints['right_eye'][0] + x1, keypoints['right_eye'][1] + y1],
                    [keypoints['nose'][0] + x1, keypoints['nose'][1] + y1],
                    [keypoints['mouth_left'][0] + x1, keypoints['mouth_left'][1] + y1],
                    [keypoints['mouth_right'][0] + x1, keypoints['mouth_right'][1] + y1]
                ], dtype=np.float32)

                for (x, y) in lmk:
                    cv2.circle(annotated, (int(x), int(y)), 2, (0, 0, 255), -1)

        return annotated, aligned_faces
    
# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     detector = FaceDetector()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated, aligned_faces = detector.detect_and_align(frame)

#         # Cửa sổ 1: khung hình có landmark
#         cv2.imshow("YOLOv8n-Face + Landmark", annotated)

#         # Cửa sổ 2: khuôn mặt đã căn chỉnh
#         if len(aligned_faces) > 0:
#             cv2.imshow("Aligned Face", aligned_faces[0])

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
