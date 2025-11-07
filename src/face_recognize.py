"""
src/face_recognize.py
Face recognition using InsightFace ArcFace (512-D) + LinearSVC.
OPTIMIZED: Reduced lag + improved accuracy while keeping original logic.
"""

import os
import sys
import pickle
import cv2
import numpy as np
import time
import threading
from typing import List, Tuple, Optional
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insightface.app import FaceAnalysis
from src.face_detect import FaceDetector


class FaceRecognizer:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "buffalo_l",
        db_path: str = "encodings/embeddings.pkl",
        threshold: float = 0.6,
        dataset_root: str = "dataset/SinhVien",
        face_app=None,
        detector=None,
    ):
        import torch

        # Auto select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.db_path = db_path
        self.threshold = max(0.6, threshold)
        self.dataset_root = dataset_root
        self.lock = threading.Lock()  # thread-safe

        print(f"[INIT] ArcFace ({model_name}) on {self.device}")

        # Reuse global face_app to avoid reload cost
        if face_app is not None:
            self.face_app = face_app
        else:
            ctx_id = 0 if self.device == "cuda" else -1
            self.face_app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"]
            )
            self.face_app.prepare(ctx_id=ctx_id, det_size=(160, 160))

        # Load or create FaceDetector
        if detector is not None:
            self.detector = detector
        else:
            yolo_path = os.path.join("models", "yolov11n-face.pt")
            if not os.path.exists(yolo_path):
                raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
            self.detector = FaceDetector(device=self.device, yolo_model_path=yolo_path)

        # DB
        self.labels, self.embeddings = [], np.empty((0, 512), np.float32)
        self.label_encoder = LabelEncoder()
        self.classifier: Optional[LinearSVC] = None

        # Cache for repeated faces
        self._embedding_cache = {}
        self._cache_expire = 3.0  # seconds per face cache

        if os.path.exists(self.db_path):
            self._load_db()
        else:
            print("[INFO] No DB found -> building new one...")
            self.build_embeddings(self.dataset_root)

        print(f"[READY] {len(self.labels)} persons | device={self.device.upper()} | Classifier={'OK' if self.classifier else 'None'}")

    # ============================================================
    # Internal Helpers
    # ============================================================
    def _load_db(self):
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.labels = list(data.get("labels", []))
            self.embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
            self.label_encoder = data.get("label_encoder", LabelEncoder())
            self.classifier = data.get("classifier", None)
            print(f"[LOAD] {len(self.labels)} entries loaded from {self.db_path}")
        except Exception as e:
            print(f"[WARN] Failed to load DB: {e}")

    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "labels": self.labels,
                "embeddings": self.embeddings,
                "label_encoder": self.label_encoder,
                "classifier": self.classifier,
            }, f)
        print(f"[SAVE] DB saved -> {self.db_path}")

    # ============================================================
    # Embedding Extraction (Optimized)
    # ============================================================
    def _extract_embedding(self, img_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_112, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb)
        if not faces:
            return np.zeros(512, np.float32)

        emb = faces[0].embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-6
        return emb

    def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
        if aligned:
            return self._extract_embedding(img_bgr)

        _, aligned_faces = self.detector.detect_and_align(img_bgr, imgsz=256)
        if not aligned_faces:
            return np.zeros(512, np.float32)
        face_img = cv2.resize(aligned_faces[0][0], (112, 112))
        return self._extract_embedding(face_img)

    # ============================================================
    # DB Build & Train
    # ============================================================
    def build_embeddings(self, dataset_root: Optional[str] = None, max_images_per_student: int = 10):
        root = dataset_root or self.dataset_root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found: {root}")

        print(f"[BUILD] Building DB from {root}")
        labels, vecs, failed = [], [], 0

        for person in sorted(os.listdir(root)):
            pdir = os.path.join(root, person)
            if not os.path.isdir(pdir):
                continue
            files = [os.path.join(pdir, f) for f in os.listdir(pdir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_images_per_student]
            for path in files:
                img = cv2.imread(path)
                if img is None:
                    failed += 1
                    continue
                emb = self.extract_embedding(img)
                if np.any(emb):
                    labels.append(person)
                    vecs.append(emb)
                else:
                    failed += 1

        if not vecs:
            print("[ERROR] No embeddings built.")
            return

        self.labels = labels
        self.embeddings = np.vstack(vecs).astype(np.float32)
        y = self.label_encoder.fit_transform(self.labels)

        # Train classifier (optimized)
        self.classifier = LinearSVC(dual=False, max_iter=1000, random_state=42)
        self.classifier.fit(self.embeddings, y)
        self._save_db()

        print(f"[DONE] Trained {len(self.labels)} samples (failed: {failed})")

    # ============================================================
    # Recognition (Optimized)
    # ============================================================
    def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
        if self.classifier is None or len(self.labels) == 0:
            return "Unknown", 0.0

        # Cache key by hash of image
        h = hash(aligned_face.tobytes()[:500])
        now = time.time()
        if h in self._embedding_cache and now - self._embedding_cache[h]["time"] < self._cache_expire:
            emb = self._embedding_cache[h]["emb"]
        else:
            emb = self._extract_embedding(aligned_face)
            self._embedding_cache[h] = {"emb": emb, "time": now}

        emb = emb.reshape(1, -1)
        scores = self.classifier.decision_function(emb)[0]
        idx = int(np.argmax(scores))
        margin = float(scores[idx] - np.partition(scores, -2)[-2]) if len(scores) > 1 else scores[idx]
        conf = 1.0 / (1.0 + np.exp(-margin))
        conf = float(np.clip(conf, 0.0, 1.0))

        label = self.label_encoder.inverse_transform([idx])[0]
        if conf < self.threshold:
            label = "Unknown"
        return label, conf

    def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float]]:
        if not aligned_faces:
            return []
        if self.classifier is None or len(self.labels) == 0:
            return [("Unknown", 0.0)] * len(aligned_faces)

        # Batch embeddings (fast path)
        embs = []
        for face, _ in aligned_faces:
            h = hash(face.tobytes()[:500])
            now = time.time()
            if h in self._embedding_cache and now - self._embedding_cache[h]["time"] < self._cache_expire:
                emb = self._embedding_cache[h]["emb"]
            else:
                emb = self._extract_embedding(face)
                self._embedding_cache[h] = {"emb": emb, "time": now}
            embs.append(emb)

        embs = np.vstack(embs).astype(np.float32)
        scores_batch = self.classifier.decision_function(embs)

        results = []
        for scores in scores_batch:
            idx = int(np.argmax(scores))
            margin = float(scores[idx] - np.partition(scores, -2)[-2]) if len(scores) > 1 else scores[idx]
            conf = 1.0 / (1.0 + np.exp(-margin))
            conf = float(np.clip(conf, 0.0, 1.0))
            label = self.label_encoder.inverse_transform([idx])[0]
            if conf < self.threshold:
                label = "Unknown"
            results.append((label, conf))
        return results
