from insightface.app import FaceAnalysis
import cv2

# Khởi tạo RetinaFace (buffalo_l có cả detector + landmark + embedding)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

print("✅ InsightFace loaded successfully!")

# Test bằng webcam (nếu muốn)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        kps = face.kps.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        for (x, y) in kps:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("RetinaFace Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
