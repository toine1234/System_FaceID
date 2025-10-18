from insightface.app import FaceAnalysis
import cv2, time

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)
print("[INFO] Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        for (x, y) in face.kps:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Landmarks Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
