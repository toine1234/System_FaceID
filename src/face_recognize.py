# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Fix import path

# import pickle
# import numpy as np
# from typing import List, Tuple, Optional
# from tqdm import tqdm
# import cv2
# import faiss

# from insightface.app import FaceAnalysis
# from src.face_detect import FaceDetector


# class FaceRecognizer:
#     def __init__(self, device: str = None, model_name: str = "buffalo_l",
#                  db_path: str = "encodings/embeddings.pkl", threshold: float = 0.6,
#                  use_faiss: bool = True, detector=None,
#                  dataset_root: str = "dataset/SinhVien"):
#         """Initialize ArcFace for 512D feature extraction."""
#         import torch
#         if device is None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             elif torch.backends.mps.is_available():
#                 device = "mps"
#             else:
#                 device = "cpu"
#         self.device = device
#         self.threshold = threshold
#         self.use_faiss = use_faiss
#         self.db_path = db_path
#         self.dataset_root = dataset_root

#         print(f"[INIT] Loading ArcFace model ({model_name}) on {device}...")

#         # Initialize ArcFace
#         ctx_id = 0 if device != "cpu" else -1
#         self.face_app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"])
#         self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

#         # Load YOLO model
#         model_path = os.path.join("models", "yolov11n-face.pt")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"YOLO model not found: {model_path}")
#         self.detector = detector if detector is not None else FaceDetector(
#             device=device, yolo_model_path=model_path
#         )

#         # Load or build embedding database
#         self.labels: List[str] = []
#         self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
#         self.index: Optional[faiss.Index] = None

#         if os.path.exists(self.db_path):
#             self._load_db()
#             if len(self.labels) == 0:
#                 print("[WARN] Database is empty -> rebuilding...")
#                 self.build_embeddings(self.dataset_root)
#         else:
#             print(f"[INFO] Database not found -> building new from {self.dataset_root}...")
#             self.build_embeddings(self.dataset_root)

#         print(f"[READY] FaceRecognizer ready with {len(self.labels)} students on {self.device.upper()}")

#     # =============================================================
#     def _load_db(self):
#         """Load embeddings database + FAISS index (if enabled)."""
#         try:
#             with open(self.db_path, 'rb') as f:
#                 data = pickle.load(f)
#             self.labels = data.get('labels', [])
#             self.embeddings = np.array(data.get('embeddings', []), dtype=np.float32)
#             if self.use_faiss and len(self.embeddings) > 0:
#                 faiss.normalize_L2(self.embeddings)
#                 self.index = faiss.IndexFlatIP(512)
#                 self.index.add(self.embeddings)
#             print(f"[INFO] Loaded database: {len(self.labels)} students from {self.db_path}")
#         except Exception as e:
#             print(f"[WARN] Failed to load database: {e}. Rebuilding...")
#             self.labels, self.embeddings = [], np.empty((0, 512), dtype=np.float32)

#     def _save_db(self):
#         """Save embeddings database to file."""
#         os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
#         with open(self.db_path, 'wb') as f:
#             pickle.dump({'labels': self.labels, 'embeddings': self.embeddings}, f)
#         print(f"[SAVE] Database saved to {self.db_path} ({len(self.labels)} entries)")

#     # =============================================================
#     def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
#         """Extract 512D embedding vector from a face image."""
#         if not aligned:
#             _, aligned_faces = self.detector.detect_and_align(img_bgr)
#             if not aligned_faces:
#                 raise ValueError("No face detected in the image.")
#             aligned_img, _ = aligned_faces[0]
#         else:
#             aligned_img = img_bgr

#         rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
#         faces = self.face_app.get(rgb_img)
#         if not faces:
#             raise ValueError("No face found in aligned image.")
#         emb = faces[0].embedding
#         emb /= np.linalg.norm(emb)
#         return emb.astype(np.float32)

#     # =============================================================
#     def build_embeddings(self, dataset_root: str = "dataset/SinhVien", aligned: bool = True,
#                          max_images_per_student: int = 5):
#         """Automatically build averaged embeddings for each student in dataset."""
#         if not os.path.exists(dataset_root):
#             raise FileNotFoundError(f"Dataset not found: {dataset_root}")

#         print(f"[BUILD] Creating embeddings from dataset: {dataset_root} ...")
#         labels, embeddings_list, failed = [], [], 0

#         for person_dir in tqdm(sorted(os.listdir(dataset_root)), desc="Building student data"):
#             person_path = os.path.join(dataset_root, person_dir)
#             if not os.path.isdir(person_path):
#                 continue

#             img_files = [os.path.join(person_path, f) for f in os.listdir(person_path)
#                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:max_images_per_student]
#             person_embeddings = []

#             for img_path in img_files:
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     failed += 1
#                     continue
#                 try:
#                     emb = self.extract_embedding(img, aligned=aligned)
#                     person_embeddings.append(emb)
#                 except Exception:
#                     failed += 1

#             if not person_embeddings:
#                 continue

#             mean_emb = np.mean(person_embeddings, axis=0)
#             mean_emb /= np.linalg.norm(mean_emb)
#             embeddings_list.append(mean_emb)
#             labels.append(person_dir)

#         if not embeddings_list:
#             print("[ERROR] No valid face data found to build database.")
#             return

#         self.labels = labels
#         self.embeddings = np.vstack(embeddings_list).astype(np.float32)

#         if self.use_faiss:
#             faiss.normalize_L2(self.embeddings)
#             self.index = faiss.IndexFlatIP(512)
#             self.index.add(self.embeddings)

#         self._save_db()
#         print(f"[DONE] Built {len(self.labels)} embeddings. (Failed images: {failed})")

#     # =============================================================
#     def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
#         """Recognize one aligned face -> (label, cosine_score)."""
#         emb = self.extract_embedding(aligned_face, aligned=True)

#         if self.index is not None:
#             faiss.normalize_L2(emb.reshape(1, -1))
#             sims, ids = self.index.search(emb.reshape(1, -1), 1)
#             sim, idx = sims[0][0], ids[0][0]
#         else:
#             sims = np.dot(self.embeddings, emb)
#             idx = np.argmax(sims)
#             sim = sims[idx]

#         label = self.labels[idx] if sim >= self.threshold else "Unknown"
#         return label, float(sim)

#     # =============================================================
#     def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float]]:
#         """Recognize multiple faces in the same frame."""
#         results = []
#         for aligned_face, _ in aligned_faces:
#             try:
#                 label, score = self.recognize(aligned_face)
#             except Exception:
#                 label, score = "Unknown", 0.0
#             results.append((label, score))

#         return results


"""
src/face_recognize.py
Face recognition using InsightFace ArcFace (512-D).
- Input: aligned_faces [(112x112 BGR, bbox)]
- Dataset: dataset/SinhVien/<StudentID_Name>/*.jpg
- DB: encodings/embeddings.pkl {labels: list[str], embeddings: float32 [N,512]}
"""

import os, sys, pickle, cv2, faiss
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fix import path
from insightface.app import FaceAnalysis
from src.face_detect import FaceDetector


class FaceRecognizer:
    def __init__(
        self,
        device: str | None = None,
        model_name: str = "buffalo_l",
        db_path: str = "encodings/embeddings.pkl",
        threshold: float = 0.6,
        use_faiss: bool = True,
        detector: Optional[FaceDetector] = None,
        dataset_root: str = "dataset/SinhVien",
    ):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.db_path, self.threshold = device, db_path, threshold
        self.use_faiss, self.dataset_root = use_faiss, dataset_root

        print(f"[INIT] ArcFace ({model_name}) on {self.device}")
        # NOTE: InsightFace FaceAnalysis chỉ hỗ trợ ctx_id=0 với CUDA; MPS/CPU -> -1
        ctx_id = 0 if self.device == "cuda" else -1
        self.face_app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"])
        self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        # Detector YOLO (phòng khi cần tự căn chỉnh ảnh chưa aligned)
        yolo_path = os.path.join("models", "yolov11n-face.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        self.detector = detector or FaceDetector(device=self.device, yolo_model_path=yolo_path)

        # DB
        self.labels: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.index: Optional[faiss.Index] = None

        if os.path.exists(self.db_path):
            self._load_db()
            if not self.labels:
                print("[WARN] Empty DB -> rebuild from dataset")
                self.build_embeddings(self.dataset_root)
        else:
            print(f"[INFO] No DB -> build from {self.dataset_root}")
            self.build_embeddings(self.dataset_root)

        print(f"[READY] {len(self.labels)} students | device={self.device.upper()} | FAISS={self.index is not None}")

    # -------------------- DB I/O --------------------
    def _ensure_faiss(self):
        if not self.use_faiss or self.embeddings.size == 0:
            self.index = None
            return
        embs = self.embeddings.copy()
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def _load_db(self):
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.labels = list(data.get("labels", []))
            embs = np.asarray(data.get("embeddings", []), dtype=np.float32)
            # đảm bảo đã chuẩn hóa cho cả trường hợp không dùng FAISS
            self.embeddings = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-6, None) if len(embs) else embs
            self._ensure_faiss()
            print(f"[INFO] Loaded DB: {len(self.labels)} entries from {self.db_path}")
        except Exception as e:
            print(f"[WARN] Failed to load DB: {e}. Resetting.")
            self.labels, self.embeddings, self.index = [], np.empty((0, 512), np.float32), None

    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({"labels": self.labels, "embeddings": self.embeddings}, f)
        print(f"[SAVE] DB -> {self.db_path} ({len(self.labels)} entries)")

    # -------------------- Embedding --------------------
    def _embed_from_aligned(self, img_bgr_112: np.ndarray) -> np.ndarray:
        """Nhận 512-D embedding từ ảnh 112x112 BGR đã align."""
        # FaceAnalysis.get sẽ tự detect; với ảnh 112x112 khuôn mặt, detect rất nhanh
        rgb = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb)
        if not faces:
            raise ValueError("No face found in aligned image.")
        emb = faces[0].embedding.astype(np.float32)
        return emb / np.linalg.norm(emb)  # L2-norm

    def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
        """Trả embedding 512-D từ ảnh; nếu chưa aligned sẽ tự detect+align khuôn mặt đầu tiên."""
        if aligned:
            return self._embed_from_aligned(img_bgr)
        _, aligned_faces = self.detector.detect_and_align(img_bgr)
        if not aligned_faces:
            raise ValueError("No face detected.")
        return self._embed_from_aligned(aligned_faces[0][0])

    # -------------------- Build DB --------------------
    def build_embeddings(self, dataset_root: str | None = None, aligned: bool = True, max_images_per_student: int = 5):
        root = dataset_root or self.dataset_root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found: {root}")

        print(f"[BUILD] From dataset: {root}")
        labels, vecs, failed = [], [], 0

        for person_dir in tqdm(sorted(os.listdir(root)), desc="Students"):
            pdir = os.path.join(root, person_dir)
            if not os.path.isdir(pdir):
                continue
            img_files = [os.path.join(pdir, f) for f in os.listdir(pdir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_images_per_student]
            emb_list = []
            for path in img_files:
                img = cv2.imread(path)
                if img is None:
                    failed += 1
                    continue
                try:
                    emb_list.append(self.extract_embedding(img, aligned=aligned))
                except Exception:
                    failed += 1
            if emb_list:
                m = np.mean(emb_list, axis=0).astype(np.float32)
                m /= np.linalg.norm(m)
                vecs.append(m)
                labels.append(person_dir)

        if not vecs:
            print("[ERROR] No valid faces to build DB.")
            return

        self.labels = labels
        self.embeddings = np.vstack(vecs).astype(np.float32)
        self._ensure_faiss()
        self._save_db()
        print(f"[DONE] Built {len(self.labels)} entries (failed images: {failed})")

    # -------------------- Recognition --------------------
    def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
        """Nhận dạng 1 khuôn mặt đã align: trả (label, cosine)."""
        if len(self.labels) == 0:
            return "Unknown", 0.0
        emb = self._embed_from_aligned(aligned_face)

        if self.index is not None:
            q = emb.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            sims, ids = self.index.search(q, 1)
            sim, idx = float(sims[0][0]), int(ids[0][0])
        else:
            sims = np.dot(self.embeddings, emb)
            idx, sim = int(np.argmax(sims)), float(sims[np.argmax(sims)])

        return (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)

    def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float]]:
        """Nhận dạng nhiều mặt trong cùng frame (tối ưu theo lô)."""
        if len(self.labels) == 0 or not aligned_faces:
            return [("Unknown", 0.0) for _ in aligned_faces]

        embs = []
        for img, _ in aligned_faces:
            try:
                embs.append(self._embed_from_aligned(img))
            except Exception:
                embs.append(None)

        results: List[Tuple[str, float]] = []
        valid_idx = [i for i, e in enumerate(embs) if e is not None]
        if not valid_idx:
            return [("Unknown", 0.0) for _ in aligned_faces]

        emb_mat = np.vstack([embs[i] for i in valid_idx]).astype(np.float32)

        if self.index is not None:
            Q = emb_mat.copy()
            faiss.normalize_L2(Q)
            sims, ids = self.index.search(Q, 1)
            top_sims = sims.ravel().tolist()
            top_ids = ids.ravel().tolist()
        else:
            sims_all = self.embeddings @ emb_mat.T   # [N,512] @ [512,M] -> [N,M]
            top_ids = np.argmax(sims_all, axis=0).tolist()
            top_sims = sims_all[top_ids, range(sims_all.shape[1])].tolist()

        res_map = {}
        for j, i in enumerate(valid_idx):
            sim, idx = float(top_sims[j]), int(top_ids[j])
            res_map[i] = (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)
        for i in range(len(aligned_faces)):
            results.append(res_map.get(i, ("Unknown", 0.0)))
        return results
