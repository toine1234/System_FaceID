from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/DELL/Desktop/System_FaceID/models/face_detector/yolov8n-face/train_results/weights/best.pt")

cap = cv2.VideoCapture(0)  # webcam mặc định

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Face Detection (YOLOv8n-Face)", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
