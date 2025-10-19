import cv2
from src.face_detect import FaceDetector

# --- Khởi tạo đối tượng ---
# Nếu bạn đang dùng RetinaFace: đảm bảo file face_detect.py có import từ insightface.app
detector = FaceDetector(
    yolo_model_path="models/yolov8n-face.pt",
    device="cpu"  # 'mps' cho Mac M1/M2/M4, hoặc 'cpu' nếu chưa hỗ trợ
)

# --- Mở webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam!")
    exit()

print("✅ Webcam đã sẵn sàng. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình.")
        break

    # --- Gọi hàm detect_and_align ---
    annotated, aligned_faces = detector.detect_and_align(frame)

    # --- Hiển thị kết quả ---
    cv2.imshow("Face Detection Test", annotated)

    # --- Thoát ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
