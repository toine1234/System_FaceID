"""
Face recognition using ArcFace (InsightFace) + SVM Classifier.
Database: encodings/embeddings.pkl (embeddings + labels + SVM model)
"""

import os
import sys
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Add project root path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from insightface.app import FaceAnalysis
from src.face_detect import FaceDetector


class FaceRecognizer:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "buffalo_l",
        db_path: str = "encodings/embeddings.pkl",
        threshold: float = 0.5,
        dataset_root: str = "dataset/SinhVien",
        face_app=None,
        detector=None
    ):
        import torch

        # --- Device selection ---
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.db_path = db_path
        self.threshold = threshold
        self.dataset_root = dataset_root

        print(f"[INIT] ArcFace ({model_name}) on {self.device}")

        # --- Load ArcFace model ---
        ctx_id = 0 if self.device == "cuda" else -1
        self.face_app = FaceAnalysis(
            name=model_name, allowed_modules=["detection", "recognition"]
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        # --- Optional YOLO detector for better alignment ---
        yolo_path = os.path.join("models", "yolov11n-face.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        self.detector = FaceDetector(device=self.device, yolo_model_path=yolo_path)

        # --- Initialize DB variables ---
        self.labels: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.label_encoder = LabelEncoder()
        self.classifier: Optional[SVC] = None

        # --- Load or build database ---
        if os.path.exists(self.db_path):
            self._load_db()
            if not self.labels:
                print("[WARN] Empty DB → rebuilding...")
                self.build_embeddings(self.dataset_root)
        else:
            print("[INFO] No DB found -> building new one...")
            self.build_embeddings(self.dataset_root)

        print(
            f"[READY] {len(self.labels)} persons | device={self.device.upper()} | Classifier={'OK' if self.classifier else 'None'}"
        )

    # -------------------- DB I/O --------------------
    def _load_db(self):
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.labels = list(data.get("labels", []))
            self.embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
            self.label_encoder = data.get("label_encoder", LabelEncoder())
            self.classifier = data.get("classifier", None)
            print(f"[LOAD] Loaded {len(self.labels)} entries from {self.db_path}")
        except Exception as e:
            print(f"[WARN] Failed to load DB: {e}")
            self.labels, self.embeddings, self.classifier = [], np.empty((0, 512), np.float32), None

    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(
                {
                    "labels": self.labels,
                    "embeddings": self.embeddings,
                    "label_encoder": self.label_encoder,
                    "classifier": self.classifier,
                },
                f,
            )
        print(f"[SAVE] DB saved to {self.db_path}")

    # -------------------- Embedding --------------------
    def _embed_from_aligned(self, img_bgr_112: np.ndarray) -> np.ndarray:
        """Extract normalized 512D embedding from an aligned face (112x112)."""
        rgb = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb)
        if not faces:
            raise ValueError("No face found in image.")
        emb = faces[0].embedding.astype(np.float32)
        return emb / np.linalg.norm(emb)

    def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
        """Extract embedding from raw or aligned image."""
        if aligned:
            return self._embed_from_aligned(img_bgr)
        _, aligned_faces = self.detector.detect_and_align(img_bgr)
        if not aligned_faces:
            raise ValueError("No face detected.")
        face_img = aligned_faces[0][0]
        face_img = cv2.resize(face_img, (112, 112))
        return self._embed_from_aligned(face_img)

    # -------------------- Build Embeddings + Train SVM --------------------
    def build_embeddings(
        self,
        dataset_root: Optional[str] = None,
        aligned: bool = True,
        max_images_per_student: int = 10,
    ):
        root = dataset_root or self.dataset_root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found: {root}")

        print(f"[BUILD] Building from dataset: {root}")
        labels, vecs, failed = [], [], 0

        for person_dir in tqdm(sorted(os.listdir(root)), desc="Students"):
            pdir = os.path.join(root, person_dir)
            if not os.path.isdir(pdir):
                continue
            img_files = [
                os.path.join(pdir, f)
                for f in os.listdir(pdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:max_images_per_student]
            for path in img_files:
                img = cv2.imread(path)
                if img is None:
                    failed += 1
                    continue
                try:
                    emb = self.extract_embedding(img, aligned=aligned)
                    vecs.append(emb)
                    labels.append(person_dir)
                except Exception:
                    failed += 1

        if not vecs:
            print("[ERROR] No valid faces found to build DB.")
            return

        self.labels = labels
        self.embeddings = np.vstack(vecs).astype(np.float32)

        # Encode labels numerically
        y = self.label_encoder.fit_transform(self.labels)

        # Train SVM classifier
        self.classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
        self.classifier.fit(self.embeddings, y)

        self._save_db()
        print(f"[DONE] Trained SVM with {len(self.labels)} samples (failed: {failed})")

    # -------------------- Recognition --------------------
    def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
        """Predict identity from an aligned face image."""
        if self.classifier is None or len(self.labels) == 0:
            return "Unknown", 0.0

        emb = self._embed_from_aligned(aligned_face).reshape(1, -1)
        probs = self.classifier.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        label = self.label_encoder.inverse_transform([idx])[0]
        if conf < self.threshold:
            return "Unknown", conf
        return label, conf

    def recognize_faces(
        self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]
    ) -> List[Tuple[str, float]]:
        """Recognize multiple faces in a frame."""
        results = []
        for face_img, _ in aligned_faces:
            try:
                label, conf = self.recognize(face_img)
            except Exception:
                label, conf = "Unknown", 0.0
            results.append((label, conf))
        return results
