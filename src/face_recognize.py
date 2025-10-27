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
        device: Optional[str] = None,
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
    def build_embeddings(self, dataset_root: Optional[str] = None, aligned: bool = True, max_images_per_student: int = 5):
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