import cv2
from src.face_detect import FaceDetector
import numpy as np

def main():
    # --- Khởi tạo detector ---
    detector = FaceDetector(
        yolo_model_path="models/yolov8n-face.pt",
        device="mps"  # dùng 'mps' cho Mac, hoặc 'cpu'/'cuda' tùy máy bạn
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được camera.")
        return

    print("[INFO] Nhấn 'q' để thoát.")
    print("[INFO] Hiển thị song song khuôn mặt gốc và khuôn mặt căn chỉnh.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không đọc được khung hình.")
            break

        annotated, aligned_faces = detector.detect_and_align(frame)

        # --- Vẽ thông tin khuôn mặt ---
        cv2.putText(annotated, f"Detected faces: {len(aligned_faces)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # --- Hiển thị song song khuôn mặt gốc & căn chỉnh ---
        display = annotated.copy()
        y_offset = 50
        x_offset = 10

        for i, (aligned_face, bbox) in enumerate(aligned_faces[:5]):
            x1, y1, x2, y2 = bbox
            face_original = frame[y1:y2, x1:x2]
            if face_original.size == 0:
                continue

            # Resize cả hai để cùng kích thước 112x112
            orig_resized = cv2.resize(face_original, (112, 112))
            aligned_resized = cv2.cvtColor(cv2.resize(aligned_face, (112, 112)), cv2.COLOR_RGB2BGR)

            # Ghép song song khuôn mặt gốc và căn chỉnh
            combined = np.hstack([orig_resized, aligned_resized])

            # Hiển thị ở góc trái
            if y_offset + 112 < display.shape[0]:
                display[y_offset:y_offset + 112, x_offset:x_offset + 224] = combined
                y_offset += 122

        cv2.imshow("Face Detection & Alignment Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
