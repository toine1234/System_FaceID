
# from ultralytics import YOLO
# from facenet_pytorch import MTCNN
# from src.alignment import norm_crop
# import cv2
# import numpy as np


# class FaceDetector:
#     def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="cpu"):
#         self.yolo = YOLO(yolo_model_path)
#         self.mtcnn = MTCNN(keep_all=True, device=device)
#         self.device = device
#         self.frame_count = 0


#     def detect_and_align(self, frame):
#         self.frame_count += 1
#         annotated = frame.copy()
#         aligned_faces = []

#         # --- Phát hiện khuôn mặt bằng YOLO ---
#         results = self.yolo(frame, imgsz=384, conf=0.5, iou=0.45, verbose=False, stream=True)
#         for r in results:
#             boxes = r.boxes.xyxy.cpu().numpy()

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 face_crop = frame[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue

#                 # --- MTCNN Landmark ---
#                 try:
#                     _, _, points = self.mtcnn.detect(face_crop, landmarks=True)
#                     if points is not None and len(points) > 0:
#                         landmark = np.array(points[0], dtype=np.float32)

#                         # --- Căn chỉnh bằng ArcFace norm_crop ---
#                         aligned = norm_crop(face_crop, landmark, image_size=112)

#                         # Vẽ bounding box & landmark lên frame
#                         cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         for (lx, ly) in landmark:
#                             cv2.circle(annotated, (int(x1 + lx), int(y1 + ly)), 2, (0, 0, 255), -1)

#                         aligned_faces.append((aligned, (x1, y1, x2, y2)))

#                     else:
#                         print("[WARN] Không tìm thấy landmark, dùng crop gốc.")
#                         aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

#                 except Exception as e:
#                     print(f"[WARN] lỗi alignment: {e}")
#                     aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

#         return annotated, aligned_faces


from ultralytics import YOLO
from insightface.app import FaceAnalysis
from src.alignment import norm_crop
import cv2, numpy as np

class FaceDetector:
    def __init__(self, yolo_model_path="models/yolov8n-face.pt", device="mps"):
        self.yolo = YOLO(yolo_model_path)
        self.device = device

        # RetinaFace nhẹ hơn (320x320)
        self.face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if device != "cpu" else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        self.frame_count = 0
        self.retina_faces = []

    def detect_and_align(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        aligned_faces = []

        # --- Chạy RetinaFace mỗi 5 frame ---
        if self.frame_count % 5 == 0:
            self.retina_faces = self.face_app.get(frame)

        retina_faces = getattr(self, "retina_faces", [])

        # --- YOLOv8n-Face ---
        results = self.yolo(frame, imgsz=256, conf=0.5, iou=0.45, verbose=False, stream=True)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                found = False
                for face in retina_faces:
                    bx1, by1, bx2, by2 = face.bbox.astype(int)
                    if (bx1 >= x1 - 15 and by1 >= y1 - 15 and bx2 <= x2 + 15 and by2 <= y2 + 15):
                        kps = face.kps.astype(np.float32)
                        aligned = norm_crop(frame, kps, image_size=112)

                        # vẽ landmark (tùy chọn)
                        for (lx, ly) in kps:
                            cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

                        aligned_faces.append((aligned, (x1, y1, x2, y2)))
                        found = True
                        break

                if not found:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        aligned_faces.append((cv2.resize(face_crop, (112, 112)), (x1, y1, x2, y2)))

        return annotated, aligned_faces


# if __name__ == "__main__":
#     detector = FaceDetector(device="mps")  # hoặc "cpu"
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         annotated, aligned_faces = detector.detect_and_align(frame)
#         cv2.imshow("Optimized YOLO + RetinaFace", annotated)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Fix import path
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import faiss
import onnxruntime as ort
from src.face_detect import FaceDetector

class ArcFaceRecognizerONNX:
    """
    ArcFace ONNX Recognizer tương thích YOLOv11n-face
    - Dataset: dataset/SinhVien/<Name>/*.jpg
    - DB: encodings/embeddings.pkl
    """

    def __init__(self,
                 onnx_path="models/arcface.onnx",
                 db_path="encodings/embeddings.pkl",
                 dataset_root="dataset/SinhVien",
                 threshold=0.6,
                 use_faiss=True,
                 detector=None):
        self.onnx_path = onnx_path
        self.db_path = db_path
        self.dataset_root = dataset_root
        self.threshold = threshold
        self.use_faiss = use_faiss

        # --- Phát hiện thiết bị (CPU dùng ONNXRuntime) ---
        self.device = "cpu"
        print(f"[INIT] ArcFaceRecognizerONNX initializing on {self.device.upper()}")

        # --- Load mô hình ONNX ---
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (112, 112)
        print(f"[READY] ArcFace ONNX loaded: {self.onnx_path}")

        # --- Load detector ---
        self.detector = detector if detector else FaceDetector(
            device=self.device,
            yolo_model_path="models/yolov11n-face.pt"
        )

        # --- Load hoặc build DB ---
        self.labels = []
        self.embeddings = np.empty((0, 512), dtype=np.float32)
        self.index = None

        if os.path.exists(self.db_path):
            self._load_db()
        else:
            print("[INFO] Database not found, building new one...")
            self.build_embeddings()

        print(f"[READY] ArcFaceRecognizerONNX ready with {len(self.labels)} embeddings")

    # ========================================================
    def preprocess(self, face_bgr):
        """Convert BGR->RGB, resize, normalize, shape=(1,3,112,112)"""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, self.input_size)
        blob = resized.astype(np.float32).transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        blob = (blob - 127.5) / 128.0
        return blob

    def extract_embedding(self, face_bgr):
        """Get 512D normalized embedding from aligned face using ONNX"""
        blob = self.preprocess(face_bgr)
        emb = self.session.run([self.output_name], {self.input_name: blob})[0]
        emb = emb.flatten().astype(np.float32)
        emb /= np.linalg.norm(emb)
        return emb

    # ========================================================
    def build_embeddings(self):
        """Build database from dataset folders"""
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_root}")

        labels, embeddings = [], []
        print(f"[BUILD] Building embeddings from {self.dataset_root} ...")

        for person in tqdm(sorted(os.listdir(self.dataset_root)), desc="Students"):
            person_path = os.path.join(self.dataset_root, person)
            if not os.path.isdir(person_path):
                continue

            person_embs = []
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    _, aligned_faces = self.detector.detect_and_align(img)
                    if not aligned_faces:
                        continue
                    face, _ = aligned_faces[0]
                    emb = self.extract_embedding(face)
                    person_embs.append(emb)
                except Exception:
                    continue

            if person_embs:
                mean_emb = np.mean(person_embs, axis=0)
                mean_emb /= np.linalg.norm(mean_emb)
                embeddings.append(mean_emb)
                labels.append(person)

        if not embeddings:
            raise RuntimeError("No valid face embeddings were built!")

        self.labels = labels
        self.embeddings = np.vstack(embeddings).astype(np.float32)

        if self.use_faiss:
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(512)
            self.index.add(self.embeddings)

        self._save_db()
        print(f"[DONE] Built embeddings for {len(self.labels)} students.")

    # ========================================================
    def recognize(self, face_bgr):
        """Recognize a single face -> (label, score)"""
        emb = self.extract_embedding(face_bgr)
        if self.index is not None:
            faiss.normalize_L2(emb.reshape(1, -1))
            sims, ids = self.index.search(emb.reshape(1, -1), 1)
            sim, idx = sims[0][0], ids[0][0]
        else:
            sims = np.dot(self.embeddings, emb)
            idx = np.argmax(sims)
            sim = sims[idx]

        label = self.labels[idx] if sim >= self.threshold else "Unknown"
        return label, float(sim)

    def recognize_faces(self, aligned_faces):
        """Recognize multiple aligned faces in frame"""
        results = []
        for face_bgr, _ in aligned_faces:
            try:
                label, score = self.recognize(face_bgr)
            except Exception:
                label, score = "Unknown", 0.0
            results.append((label, score))
        return results

    # ========================================================
    def _load_db(self):
        """Load DB from pickle"""
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        self.labels = data["labels"]
        self.embeddings = np.array(data["embeddings"], dtype=np.float32)

        if self.use_faiss:
            faiss.normalize_L2(self.embeddings)
            self.index = faiss.IndexFlatIP(512)
            self.index.add(self.embeddings)

        print(f"[LOAD] Loaded {len(self.labels)} embeddings from {self.db_path}")

    def _save_db(self):
        """Save DB to pickle"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({"labels": self.labels, "embeddings": self.embeddings}, f)
        print(f"[SAVE] Database saved -> {self.db_path}")
