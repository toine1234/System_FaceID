import os, sys, pickle, cv2, numpy as np, time, threading, glob
from typing import List, Tuple, Optional
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from face_detect import FaceDetector

class FaceRecognizer:
    def __init__(self, device: Optional[str]=None, model_name: str="buffalo_l",
                 db_path: Optional[str]=None, embeddings_dir: str="dataset_embeddings/",
                 classifier_path: Optional[str]=None, threshold: float=0.6,
                 dataset_root: str="dataset/SinhVien", face_app=None, detector=None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.device, self.embeddings_dir = device, embeddings_dir
        self.classifier_path = classifier_path or os.path.join(embeddings_dir, "classifier.pkl")
        self.threshold, self.dataset_root = max(0.6, threshold), dataset_root
        self.lock = threading.Lock()
        
        print(f"[INIT] ArcFace ({model_name}) on {self.device}")
        
        self.face_app = face_app or self._init_face_app(model_name)
        self.detector = detector or self._init_detector()
        
        self.labels = []
        self.embeddings = np.empty((0, 512), np.float32)
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self._embedding_cache = {}
        
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        if self._has_embeddings():
            self._load_embeddings_new()
            self._load_classifier()
        elif os.path.exists(self.dataset_root):
            print("[INFO] Building embeddings from dataset...")
            self.build_embeddings(self.dataset_root)
        else:
            print(f"[WARN] No embeddings or dataset found")
        
        print(f"[READY] {len(self.labels)} persons | device={self.device.upper()} | Classifier={'OK' if self.classifier else 'None'}")

    def _init_face_app(self, model_name):
        ctx_id = 0 if self.device == "cuda" else -1
        app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=ctx_id, det_size=(160, 160))
        return app

    def _init_detector(self):
        yolo_path = os.path.join("models", "yolov11n-face.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        return FaceDetector(device=self.device, yolo_model_path=yolo_path)

    def _load_db(self):
        if self._has_embeddings():
            self._load_embeddings_new()
            self._load_classifier()

    def _has_embeddings(self) -> bool:
        return len(glob.glob(os.path.join(self.embeddings_dir, "*_embedding.pkl"))) > 0

    def _load_embeddings_new(self):
        files = sorted(glob.glob(os.path.join(self.embeddings_dir, "*_embedding.pkl")))
        labels, embs = [], []
        
        for fpath in files:
            person_name = os.path.basename(fpath).replace("_embedding.pkl", "")
            try:
                with open(fpath, "rb") as f:
                    emb = pickle.load(f)
                emb_arr = np.asarray(emb["embedding"] if isinstance(emb, dict) and "embedding" in emb else emb, dtype=np.float32)
                if emb_arr.shape == (512,):
                    labels.append(person_name)
                    embs.append(emb_arr)
            except Exception as e:
                print(f"[WARN] Failed to load {fpath}: {e}")
        
        if embs:
            self.labels = labels
            self.embeddings = np.vstack(embs).astype(np.float32)
            self.label_encoder = LabelEncoder().fit(self.labels)
            unique = len(set(self.labels))
            print(f"[LOAD] {len(self.labels)} embeddings | {unique} unique persons")
        else:
            self.labels, self.embeddings = [], np.empty((0, 512), np.float32)

    def _load_classifier(self):
        if os.path.exists(self.classifier_path):
            try:
                with open(self.classifier_path, "rb") as f:
                    self.classifier = pickle.load(f)
                print("[LOAD] Classifier loaded")
            except Exception as e:
                print(f"[WARN] Failed to load classifier: {e}")
                self._train_classifier()
        else:
            self._train_classifier()

    def _train_classifier(self):
        if len(self.labels) == 0 or self.embeddings.shape[0] == 0:
            self.classifier = None
            return
        
        unique = len(set(self.labels))
        print(f"[TRAIN] Labels: {len(self.labels)} | Unique: {unique}")
        
        if unique < 2:
            print(f"[TRAIN] Only {unique} class → cosine similarity (fallback)")
            self.classifier = None
            return
        
        y = self.label_encoder.transform(self.labels)
        self.classifier = LinearSVC(dual=False, max_iter=1000, random_state=42)
        self.classifier.fit(self.embeddings, y)
        self._save_classifier()
        print(f"[TRAIN] SVM trained on {unique} classes")

    def _save_classifier(self):
        if self.classifier_path:
            os.makedirs(os.path.dirname(self.classifier_path) or ".", exist_ok=True)
            with open(self.classifier_path, "wb") as f:
                pickle.dump(self.classifier, f)

    def _extract_embedding(self, img_112: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_112, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb)
        if not faces:
            return np.zeros(512, np.float32)
        emb = faces[0].embedding.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def extract_embedding(self, img_bgr: np.ndarray, aligned: bool=True) -> np.ndarray:
        if aligned:
            return self._extract_embedding(img_bgr)
        _, aligned_faces = self.detector.detect_and_align(img_bgr)
        if not aligned_faces:
            return np.zeros(512, np.float32)
        return self._extract_embedding(cv2.resize(aligned_faces[0][0], (112, 112)))

    def build_embeddings(self, dataset_root: Optional[str]=None, max_images: int=10, train_classifier: bool=True):
        root = dataset_root or self.dataset_root
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset not found: {root}")
        
        print(f"[BUILD] Building from {root}")
        failed, created = 0, 0
        
        for person in sorted(os.listdir(root)):
            pdir = os.path.join(root, person)
            if not os.path.isdir(pdir):
                continue
            
            files = [os.path.join(pdir, f) for f in os.listdir(pdir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_images]
            embs = []
            
            for path in files:
                img = cv2.imread(path)
                if img is None:
                    failed += 1
                    continue
                emb = self.extract_embedding(img)
                if np.any(emb):
                    embs.append(emb)
                else:
                    failed += 1
            
            if embs:
                avg_emb = np.mean(embs, axis=0).astype(np.float32)
                avg_emb /= np.linalg.norm(avg_emb) + 1e-6
                pkl_path = os.path.join(self.embeddings_dir, f"{person}_embedding.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(avg_emb, f)
                created += 1
        
        self._load_embeddings_new()
        if train_classifier:
            self._train_classifier()
        print(f"[DONE] {created} files | failed images: {failed}")

    def _recognize_batch(self, embs: np.ndarray) -> List[Tuple[str, float]]:
        if self.classifier is None:
            sims = np.dot(embs, self.embeddings.T)
            results = []
            for sim_row in sims:
                idx = np.argmax(sim_row)
                sim = float(sim_row[idx])
                label = self.labels[idx] if sim >= self.threshold else "Unknown"
                results.append((label, sim if label != "Unknown" else 0.0))
            return results
        
        try:
            scores_batch = self.classifier.decision_function(embs)
        except:
            return [("Unknown", 0.0)] * len(embs)
        
        results = []
        for scores in scores_batch:
            idx = int(np.argmax(scores))
            margin = scores[idx] - (np.partition(scores, -2)[-2] if len(scores) > 1 else scores[idx])
            conf = float(np.clip(1.0 / (1.0 + np.exp(-margin)), 0.0, 1.0))
            label = self.label_encoder.inverse_transform([idx])[0] if conf >= self.threshold else "Unknown"
            results.append((label, conf if label != "Unknown" else 0.0))
        return results

    def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
        if len(self.labels) == 0:
            return "Unknown", 0.0
        
        h = hash(aligned_face.tobytes()[:500])
        now = time.time()
        if h in self._embedding_cache and now - self._embedding_cache[h]["time"] < 3.0:
            emb = self._embedding_cache[h]["emb"]
        else:
            emb = self._extract_embedding(aligned_face)
            self._embedding_cache[h] = {"emb": emb, "time": now}
        
        result = self._recognize_batch(emb.reshape(1, -1))[0]
        return result

    def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int,int,int,int]]]) -> List[Tuple[str, float]]:
        if not aligned_faces or len(self.labels) == 0:
            return [("Unknown", 0.0)] * len(aligned_faces)
        
        embs = []
        for face, _ in aligned_faces:
            h = hash(face.tobytes()[:500])
            now = time.time()
            if h in self._embedding_cache and now - self._embedding_cache[h]["time"] < 3.0:
                emb = self._embedding_cache[h]["emb"]
            else:
                emb = self._extract_embedding(face)
                self._embedding_cache[h] = {"emb": emb, "time": now}
            embs.append(emb)
        
        return self._recognize_batch(np.vstack(embs).astype(np.float32))

    def verify_user_embedding(self, user_name: str, img_bgr: np.ndarray) -> Tuple[bool, float]:
        pkl_path = os.path.join(self.embeddings_dir, f"{user_name}_embedding.pkl")
        if not os.path.exists(pkl_path):
            return False, 0.0
        
        try:
            with open(pkl_path, "rb") as f:
                user_emb = np.asarray(pickle.load(f), dtype=np.float32)
            if user_emb.shape != (512,):
                return False, 0.0
        except:
            return False, 0.0
        
        new_emb = self.extract_embedding(img_bgr, aligned=False)
        if np.all(new_emb == 0):
            return False, 0.0
        
        sim = float(np.dot(user_emb, new_emb))
        return sim >= self.threshold, sim

    def save_user_embedding(self, user_name: str, img_bgr: np.ndarray, retrain: bool=True):
        emb = self.extract_embedding(img_bgr, aligned=False)
        if np.all(emb == 0):
            raise ValueError("No face detected.")
        
        pkl_path = os.path.join(self.embeddings_dir, f"{user_name}_embedding.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(emb, f)
        
        self._load_embeddings_new()
        if retrain:
            self._train_classifier()
