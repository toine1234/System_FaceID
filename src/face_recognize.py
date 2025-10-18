"""
src/face_recognize.py
---------------------
Trích xuất embedding khuôn mặt và nhận dạng sinh viên
Cấu trúc dataset:
    dataset/SinhVien/<MaSV_TenSinhVien>/*.jpg
"""

import os
import pickle
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1


class FaceRecognizer:
    def __init__(self, device: str = None, pretrained_model: str = "vggface2"):
        """
        Khởi tạo mô hình trích xuất embedding FaceNet (InceptionResnetV1)
        - device: 'cpu' hoặc 'cuda'
        - pretrained_model: 'vggface2' hoặc 'casia-webface'
        """
        if device is None:
            device = "mps" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = InceptionResnetV1(pretrained=pretrained_model).eval().to(self.device)
        print(f"[INFO] FaceRecognizer loaded ({pretrained_model}) on {self.device}")

    # ============================================================
    # 1️⃣ TRÍCH XUẤT EMBEDDING
    # ============================================================
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Chuẩn hoá ảnh (resize, scale, tensor)."""
        img = image.resize((160, 160))
        np_img = np.asarray(img).astype(np.float32) / 255.0
        np_img = (np_img - 0.5) / 0.5  # scale [-1,1]
        np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(np_img).unsqueeze(0).to(self.device)
        return tensor

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Trích xuất embedding (512D) từ ảnh khuôn mặt BGR numpy.
        """
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.preprocess_image(pil_img)
        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy().reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb.astype(np.float32)

    # ============================================================
    # 2️⃣ XÂY DỰNG FILE embeddings.pkl
    # ============================================================
    def build_embeddings(self, dataset_root="dataset/SinhVien", save_path="encodings/embeddings.pkl"):
        """
        Duyệt dataset/SinhVien/<label>/ để sinh embedding cho từng ảnh.
        """
        labels = []
        embeddings = []

        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {dataset_root}")

        print("[INFO] Bắt đầu sinh embeddings từ thư mục:", dataset_root)
        for label in sorted(os.listdir(dataset_root)):
            person_dir = os.path.join(dataset_root, label)
            if not os.path.isdir(person_dir):
                continue

            img_files = [
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not img_files:
                continue

            print(f" → {label}: {len(img_files)} ảnh")
            for img_path in tqdm(img_files, desc=f"   Embedding {label}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = self.preprocess_image(img)
                    with torch.no_grad():
                        emb = self.model(tensor).cpu().numpy().reshape(-1)
                    emb = emb / (np.linalg.norm(emb) + 1e-10)
                    embeddings.append(emb.astype(np.float32))
                    labels.append(label)
                except Exception as e:
                    print(f"[WARN] Lỗi xử lý {img_path}: {e}")

        if not embeddings:
            raise RuntimeError("Không có ảnh hợp lệ trong dataset để sinh embeddings.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {"labels": labels, "embeddings": np.array(embeddings, dtype=np.float32)}
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"[DONE] Đã lưu embeddings: {save_path}")
        print(f"    → Tổng số khuôn mặt: {len(labels)}")
        print("[DEBUG] Danh sách nhãn sinh viên:", sorted(set(labels)))

    # ============================================================
    # 3️⃣ NHẬN DẠNG KHUÔN MẶT
    # ============================================================
    @staticmethod
    def load_embeddings(path="encodings/embeddings.pkl"):
        """Đọc file embeddings.pkl"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Tính độ tương đồng cosine giữa 2 vector."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def recognize_face(self, emb: np.ndarray, embeddings_path="encodings/embeddings.pkl", threshold: float = 0.7):
        """
        Nhận dạng khuôn mặt từ embedding so với cơ sở dữ liệu sinh viên.
        Trả về: (label, score)
        """
        data = self.load_embeddings(embeddings_path)
        labels = np.array(data["labels"])
        db_embs = np.array(data["embeddings"], dtype=np.float32)

        sims = np.dot(db_embs, emb)  # cosine similarity
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            return labels[best_idx], best_score
        else:
            return "Unknown", best_score

#chạy dự đoán so sánh ảnh với ảnh có trong chương trình
""" if __name__ == "__main__":
    import cv2

    recognizer = FaceRecognizer()

    # Tải file embedding
    test_path = "dataset/SinhVien/DihTris/DihTris.jpg"  # ảnh test thật sự
    img = cv2.imread(test_path)

    emb = recognizer.get_embedding(img)
    label, score = recognizer.recognize_face(emb, "encodings/embeddings.pkl", threshold=0.7)

    print(f"[RESULT] Ảnh {test_path} → Dự đoán: {label} ({score:.2f})") """
    
#mỗi lần có ảnh mới phải load lại embedding 1 lần   
if __name__ == "__main__":
    fr = FaceRecognizer(device="cpu")
    fr.build_embeddings(
        dataset_root="dataset/SinhVien",
        save_path="encodings/embeddings.pkl"
    )    
