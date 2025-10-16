# src/test_alignment.py
import cv2
from src.face_detect import FaceDetector

# Khởi tạo detector có YOLOv8-Face + MTCNN
detector = FaceDetector(model_path="models/face_detector/train_results/weights/best.pt")

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame từ webcam.")
        break

    # Phát hiện + căn chỉnh khuôn mặt
    frame, aligned_faces = detector.detect_and_align(frame)

    # Hiển thị khuôn mặt đã căn chỉnh trong cửa sổ nhỏ
    for i, face in enumerate(aligned_faces):
        face_resized = cv2.resize(face, (150, 150))
        cv2.imshow(f"Aligned Face {i+1}", face_resized)

    cv2.imshow("Face Detection + Alignment", frame)

    # Bấm ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()





# # Face Detection + Face Alignment (MTCNN)
# from ultralytics import YOLO
# from mtcnn.mtcnn import MTCNN
# import cv2
# import numpy as np

# class FaceDetector:
#     def __init__(self, 
#                  yolo_model_path="models/face_detector/train_results/weights/best.pt"):
#         self.model = YOLO(yolo_model_path)
#         self.aligner = MTCNN()

#     def detect_faces(self, frame):
#         """
#         Phát hiện khuôn mặt và trả về danh sách bounding box (x1, y1, x2, y2)
#         """
#         results = self.model(frame, stream=True)
#         annotated_frame = frame.copy()

#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(annotated_frame, "Face", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         return annotated_frame

#     def align_face(self, face_img):
#         """
#         Căn chỉnh khuôn mặt bằng MTCNN (xoay thẳng 2 mắt)
#         """
#         detections = self.aligner.detect_faces(face_img)
#         if len(detections) == 0:
#             return face_img

#         keypoints = detections[0]['keypoints']
#         left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']

#         # Tính góc lệch giữa 2 mắt
#         dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
#         angle = np.degrees(np.arctan2(dy, dx))

#         # Xoay ảnh để 2 mắt nằm ngang
#         center = ((face_img.shape[1] // 2), (face_img.shape[0] // 2))
#         M = cv2.getRotationMatrix2D(center, angle, 1)
#         aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
#         return aligned

#     def detect_and_align(self, frame):
#         """
#         Phát hiện + căn chỉnh tất cả khuôn mặt trong frame
#         Trả về list khuôn mặt đã căn chỉnh + frame có bounding box
#         """
#         aligned_faces = []
#         boxes = self.detect_faces(frame)

#         for (x1, y1, x2, y2) in boxes:
#             face_crop = frame[y1:y2, x1:x2]
#             aligned_face = self.align_face(face_crop)
#             aligned_faces.append(aligned_face)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, "Face", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         return frame, aligned_faces
    