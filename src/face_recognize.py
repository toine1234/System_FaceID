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
import torch
import logging # Thêm logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fix import path
from insightface.app import FaceAnalysis
# from src.face_detect import FaceDetector # Tạm thời không cần import vòng
# from insightface.app import FaceAnalysis # Bị lặp


class FaceRecognizer:
    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "buffalo_l",
        face_app: Optional[FaceAnalysis] = None,
        db_path: str = "encodings/embeddings.pkl",
        threshold: float = 0.6,
        use_faiss: bool = True,
        detector: Optional['FaceDetector'] = None, # Dùng 'FaceDetector' trong ngoặc
        dataset_root: str = "dataset/SinhVien",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.db_path, self.threshold = device, db_path, threshold
        self.use_faiss, self.dataset_root = use_faiss, dataset_root

        print(f"[INIT] ArcFace ({model_name}) on {self.device}")

        if face_app:
            self.face_app = face_app
            print("[INFO] ArcFace (Shared) ready")
        else:
            print("[WARN] Creating new FaceAnalysis for Recognizer.")
            ctx_id = 0 if self.device == "cuda" else -1
            self.face_app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"])
            self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

        # ================================================================
        # TỐI ƯU: Cache lại mô hình recognition để gọi trực tiếp
        # ================================================================
        if 'recognition' not in self.face_app.models:
            logging.error("FaceAnalysis object must be initialized with 'recognition' module.")
            raise RuntimeError("FaceAnalysis object must be initialized with 'recognition'.")
        self.recognition_model = self.face_app.models['recognition']
        # ================================================================

        # Detector YOLO (chỉ load khi thực sự cần, ví dụ: build_embeddings)
        self.detector = detector
        self._detector_loader = lambda: FaceDetector( # Sử dụng lambda để trì hoãn việc load
            device=self.device, 
            yolo_model_path=os.path.join("models", "yolov11n-face.pt"),
            face_app=self.face_app
        )

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
            norm_embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-6, None)
            self.embeddings = norm_embs if len(embs) else np.empty((0, 512), dtype=np.float32)
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
        """
        Nhận 512-D embedding từ ảnh 112x112 BGR đã align.
        (TỐI ƯU: Gọi thẳng get_feat thay vì get)
        """

        # Mô hình buffalo_l recognition nhận ảnh BGR 112x112
        img_rgb_112 = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
        emb = self.recognition_model.get_feat(img_bgr_112)
        
        # Chuẩn hóa L2
        return emb.astype(np.float32) / np.linalg.norm(emb)
        # ================================================================

    def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
        """Trả embedding 512-D từ ảnh; nếu chưa aligned sẽ tự detect+align khuôn mặt đầu tiên."""
        if aligned:
            return self._embed_from_aligned(img_bgr)
        
        # Tải detector nếu đây là lần đầu gọi
        if self.detector is None:
            from src.face_detect import FaceDetector # Import tại chỗ
            print("[INFO] Lazy loading detector for embedding extraction...")
            self.detector = self._detector_loader()
            
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
                except Exception as e:
                    logging.warning(f"Failed to embed {path}: {e}")
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
        
        # Hàm này giờ đã rất nhanh
        emb = self._embed_from_aligned(aligned_face)

        if self.index is not None:
            q = emb.reshape(1, -1).copy()
            # faiss.normalize_L2(q) # Không cần vì emb đã được chuẩn hóa
            sims, ids = self.index.search(q, 1)
            sim, idx = float(sims[0][0]), int(ids[0][0])
        else:
            sims = np.dot(self.embeddings, emb) # Cả 2 vector đều đã được chuẩn hóa
            idx, sim = int(np.argmax(sims)), float(sims[np.argmax(sims)])

        return (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)

    def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float]]:
        """Nhận dạng nhiều mặt trong cùng frame (tối ưu theo lô)."""
        if len(self.labels) == 0 or not aligned_faces:
            return [("Unknown", 0.0) for _ in aligned_faces]

        # 1. Trích xuất embedding theo lô (giờ đã rất nhanh)
        embs = []
        for img, _ in aligned_faces:
            try:
                embs.append(self._embed_from_aligned(img))
            except Exception:
                embs.append(None) # Thêm None nếu trích xuất thất bại

        results: List[Tuple[str, float]] = []
        valid_idx = [i for i, e in enumerate(embs) if e is not None]
        if not valid_idx: # Nếu không có embedding nào hợp lệ
            return [("Unknown", 0.0) for _ in aligned_faces]

        # 2. Tạo ma trận embedding
        emb_mat = np.vstack([embs[i] for i in valid_idx]).astype(np.float32)
        # Không cần chuẩn hóa L2 nữa vì _embed_from_aligned đã làm

        # 3. Tìm kiếm theo lô
        if self.index is not None:
            # Q = emb_mat.copy()
            # faiss.normalize_L2(Q) # Không cần
            sims, ids = self.index.search(emb_mat, 1)
            top_sims = sims.ravel().tolist()
            top_ids = ids.ravel().tolist()
        else:
            # [N, 512] @ [512, M] -> [N, M] (N: db, M: query)
            sims_all = self.embeddings @ emb_mat.T   
            top_ids = np.argmax(sims_all, axis=0).tolist()
            top_sims = sims_all[top_ids, range(sims_all.shape[1])].tolist()

        # 4. Map kết quả
        res_map = {}
        for j, i in enumerate(valid_idx):
            sim, idx = float(top_sims[j]), int(top_ids[j])
            res_map[i] = (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)
        
        # Trả về kết quả theo đúng thứ tự ban đầu
        for i in range(len(aligned_faces)):
            results.append(res_map.get(i, ("Unknown", 0.0))) # Dùng 0.0 cho các mặt bị lỗi
        return results
