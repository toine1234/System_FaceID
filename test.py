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
