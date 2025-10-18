import cv2
from test import FaceDetector
from src.face_recognize import FaceRecognizer

def main():
    # 1️⃣ Khởi tạo detector & recognizer
    detector = FaceDetector(device="mps")  # hoặc "cuda", "cpu"
    recognizer = FaceRecognizer(device="mps")

    # 2️⃣ Mở webcam
    cap = cv2.VideoCapture(0)
    print("[INFO] Đang mở webcam... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3️⃣ Phát hiện + căn chỉnh
        annotated, aligned_faces = detector.detect_and_align(frame)

        # 4️⃣ Nhận dạng từng khuôn mặt
        for face_img, (x1, y1, x2, y2) in aligned_faces:
            emb = recognizer.get_embedding(face_img)
            label, score = recognizer.recognize_face(emb, threshold=0.7)
            cv2.putText(annotated, f"{label} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Face Recognition - YOLO + RetinaFace", annotated)

        # 5️⃣ Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
