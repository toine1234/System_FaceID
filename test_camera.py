import cv2
from src.face_detect import FaceDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector("models/yolov8n-face.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện + căn chỉnh
        annotated_frame, aligned_faces = detector.detect_and_align(frame)

        # Hiển thị ảnh webcam có bounding boxes
        cv2.imshow("Face Detection", annotated_frame)

        # Hiển thị từng khuôn mặt đã căn chỉnh
        for idx, face in enumerate(aligned_faces):
            cv2.imshow(f"Aligned_{idx}", face)

        # Nhấn Q để thoát
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
