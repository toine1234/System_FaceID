from ultralytics import YOLO
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

class FaceDetector:
    def __init__(self, model_path="models/face_detector/yolov8n-face/train_results/weights/best.pt"):
        self.model = YOLO(model_path)
        self.aligner = MTCNN()

    def detect_faces(self, frame):
        boxes = []
        results = self.model(frame, stream = True)
        for r in results:
            boxes = r.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                boxes.append((x1, y1, x2, y2))
        return boxes
    
    def align_face(self, face_img):
        detections = self.aligner.detect_faces(face_img)
        if len(detections) == 0:
            return face_img
        keypoints = detections[0]['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        # Góc lệch giữa 2 mắt
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Xoay ảnh để 2 mắt nằm ngang
        center = ((face_img.shape[1] // 2), (face_img.shape[0] // 2))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
        return aligned_face
    
    def detec_and_align(self, frame):
        boxes = self.detect_faces(frame)
        aligned_faces = []
        for (x1, y1, x2, y2) in boxes:
            face_crop = frame[y1:y2, x1:x2]
            aligned_face = self.align_face(face_crop)
            aligned_faces.append(aligned_face)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return aligned_faces, frame
