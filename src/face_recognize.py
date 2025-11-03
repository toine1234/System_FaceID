import os, pickle, cv2, logging
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import torch
from insightface.app import FaceAnalysis

class FaceRecognizer:
    def __init__(self, device=None, model="buffalo_l", db_path="encodings/embeddings.pkl", 
                 threshold=0.5, dataset_root="dataset/SinhVien", face_app=None, detector=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.db_path, self.threshold, self.dataset_root = db_path, threshold, dataset_root
        self.detector = detector
        
        self.face_app = face_app or self._init_face_app(model)
        self.recognition_model = self.face_app.models["recognition"]
        
        self.labels = []
        self.embeddings = np.empty((0, 512), dtype=np.float32)
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        if os.path.exists(db_path):
            self._load_db()
        if len(self.labels) == 0:
            self.build_embeddings(dataset_root)
        if len(self.labels) > 0:
            self._train_svm()
    
    def _init_face_app(self, model):
        ctx_id = 0 if self.device == "cuda" else -1
        app = FaceAnalysis(name=model, allowed_modules=["recognition"])
        app.prepare(ctx_id=ctx_id, det_size=(112, 112))
        return app
    
    def _load_db(self):
        try:
            with open(self.db_path, "rb") as f:
                d = pickle.load(f)
                self.labels = d.get("labels", [])
                self.embeddings = np.asarray(d.get("embeddings", []), dtype=np.float32)
                self.classifier = d.get("classifier")
                self.scaler = d.get("scaler", StandardScaler())
                self.label_encoder = d.get("label_encoder", LabelEncoder())
        except Exception as e:
            logging.warning(f"Load failed: {e}")
    
    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "labels": self.labels,
                "embeddings": self.embeddings,
                "classifier": self.classifier,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder
            }, f)
    
    def _get_embedding(self, img_aligned_112):
        """Extract 512D L2-normalized embedding"""
        emb = self.recognition_model.get_feat(img_aligned_112)
        return emb / (np.linalg.norm(emb) + 1e-6)
    
    def build_embeddings(self, dataset_root):
        """Build embeddings DB from dataset"""
        self.labels, emb_dict = [], {}
        
        for person in tqdm(os.listdir(dataset_root), desc="Building embeddings"):
            person_path = os.path.join(dataset_root, person)
            if not os.path.isdir(person_path):
                continue
            
            embs = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    face = self.face_app.get(img)
                    if len(face) == 0:
                        continue
                    
                    # Align & extract
                    M = face[0].alignment_matrix
                    img_aligned = cv2.warpAffine(img, M, (112, 112))
                    emb = self._get_embedding(img_aligned)
                    embs.append(emb)
                except:
                    pass
            
            if embs:
                mean_emb = np.mean(np.array(embs), axis=0)
                mean_emb /= (np.linalg.norm(mean_emb) + 1e-6)
                emb_dict[person] = mean_emb
        
        if emb_dict:
            self.labels = list(emb_dict.keys())
            self.embeddings = np.array([emb_dict[l] for l in self.labels], dtype=np.float32)
            self._train_svm()
            self._save_db()
    
    def _train_svm(self):
        """Train SVM classifier"""
        if len(self.labels) < 2:
            return
        
        labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.classifier = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="auto", probability=True)
        )
        self.classifier.fit(self.embeddings, labels_encoded)
    
    def recognize_batch(self, imgs):
        """Recognize faces in batch"""
        results = []
        for img in imgs:
            faces = self.face_app.get(img)
            for face in faces:
                M = face.alignment_matrix
                aligned = cv2.warpAffine(img, M, (112, 112))
                emb = self._get_embedding(aligned)
                
                if self.classifier:
                    probs = self.classifier.predict_proba([emb])[0]
                    conf = probs.max()
                    label = self.label_encoder.inverse_transform([probs.argmax()])[0] if conf >= self.threshold else "Unknown"
                else:
                    label, conf = "Unknown", 0.0
                
                results.append({"name": label, "confidence": conf, "bbox": face.bbox})
        
        return results
    
    def recognize(self, img):
        """Single image recognition"""
        return self.recognize_batch([img])
