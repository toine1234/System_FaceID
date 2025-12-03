# import os, sys, pickle, cv2, faiss
# import numpy as np
# from typing import List, Tuple, Optional
# from tqdm import tqdm
# import torch
# import logging # Thêm logging

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fix import path
# from insightface.app import FaceAnalysis
# # from src.face_detect import FaceDetector # Tạm thời không cần import vòng
# # from insightface.app import FaceAnalysis # Bị lặp


# class FaceRecognizer:
#     def __init__(
#         self,
#         device: Optional[str] = None,
#         model_name: str = "buffalo_l",
#         face_app: Optional[FaceAnalysis] = None,
#         db_path: str = "encodings/embeddings.pkl",
#         threshold: float = 0.6,
#         use_faiss: bool = True,
#         detector: Optional['FaceDetector'] = None, # Dùng 'FaceDetector' trong ngoặc
#         dataset_root: str = "dataset/SinhVien",
#     ):
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#         self.device, self.db_path, self.threshold = device, db_path, threshold
#         self.use_faiss, self.dataset_root = use_faiss, dataset_root

#         print(f"[INIT] ArcFace ({model_name}) on {self.device}")

#         if face_app:
#             self.face_app = face_app
#             print("[INFO] ArcFace (Shared) ready")
#         else:
#             print("[WARN] Creating new FaceAnalysis for Recognizer.")
#             ctx_id = 0 if self.device == "cuda" else -1
#             self.face_app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition"])
#             self.face_app.prepare(ctx_id=ctx_id, det_size=(320, 320))

#         # ================================================================
#         # TỐI ƯU: Cache lại mô hình recognition để gọi trực tiếp
#         # ================================================================
#         if 'recognition' not in self.face_app.models:
#             logging.error("FaceAnalysis object must be initialized with 'recognition' module.")
#             raise RuntimeError("FaceAnalysis object must be initialized with 'recognition'.")
#         self.recognition_model = self.face_app.models['recognition']
#         # ================================================================

#         # Detector YOLO (chỉ load khi thực sự cần, ví dụ: build_embeddings)
#         self.detector = detector
#         self._detector_loader = lambda: FaceDetector( # Sử dụng lambda để trì hoãn việc load
#             device=self.device, 
#             yolo_model_path=os.path.join("models", "yolov11n-face.pt"),
#             face_app=self.face_app
#         )

#         # DB
#         self.labels: List[str] = []
#         self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
#         self.index: Optional[faiss.Index] = None

#         if os.path.exists(self.db_path):
#             self._load_db()
#             if not self.labels:
#                 print("[WARN] Empty DB -> rebuild from dataset")
#                 self.build_embeddings(self.dataset_root)
#         else:
#             print(f"[INFO] No DB -> build from {self.dataset_root}")
#             self.build_embeddings(self.dataset_root)

#         print(f"[READY] {len(self.labels)} students | device={self.device.upper()} | FAISS={self.index is not None}")

#     # -------------------- DB I/O --------------------
#     def _ensure_faiss(self):
#         if not self.use_faiss or self.embeddings.size == 0:
#             self.index = None
#             return
#         embs = self.embeddings.copy()
#         faiss.normalize_L2(embs)
#         self.index = faiss.IndexFlatIP(embs.shape[1])
#         self.index.add(embs)

#     def _load_db(self):
#         try:
#             with open(self.db_path, "rb") as f:
#                 data = pickle.load(f)
#             self.labels = list(data.get("labels", []))
#             embs = np.asarray(data.get("embeddings", []), dtype=np.float32)
#             # đảm bảo đã chuẩn hóa cho cả trường hợp không dùng FAISS
#             norm_embs = embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-6, None)
#             self.embeddings = norm_embs if len(embs) else np.empty((0, 512), dtype=np.float32)
#             self._ensure_faiss()
#             print(f"[INFO] Loaded DB: {len(self.labels)} entries from {self.db_path}")
#         except Exception as e:
#             print(f"[WARN] Failed to load DB: {e}. Resetting.")
#             self.labels, self.embeddings, self.index = [], np.empty((0, 512), np.float32), None

#     def _save_db(self):
#         os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
#         with open(self.db_path, "wb") as f:
#             pickle.dump({"labels": self.labels, "embeddings": self.embeddings}, f)
#         print(f"[SAVE] DB -> {self.db_path} ({len(self.labels)} entries)")

#     # -------------------- Embedding --------------------
#     def _embed_from_aligned(self, img_bgr_112: np.ndarray) -> np.ndarray:
#         """
#         Nhận 512-D embedding từ ảnh 112x112 BGR đã align.
#         (TỐI ƯU: Gọi thẳng get_feat thay vì get)
#         """

#         # Mô hình buffalo_l recognition nhận ảnh BGR 112x112
#         img_rgb_112 = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
#         emb = self.recognition_model.get_feat(img_bgr_112)
        
#         # Chuẩn hóa L2
#         return emb.astype(np.float32) / np.linalg.norm(emb)
#         # ================================================================

#     def extract_embedding(self, img_bgr: np.ndarray, aligned: bool = True) -> np.ndarray:
#         """Trả embedding 512-D từ ảnh; nếu chưa aligned sẽ tự detect+align khuôn mặt đầu tiên."""
#         if aligned:
#             return self._embed_from_aligned(img_bgr)
        
#         # Tải detector nếu đây là lần đầu gọi
#         if self.detector is None:
#             from src.face_detect import FaceDetector # Import tại chỗ
#             print("[INFO] Lazy loading detector for embedding extraction...")
#             self.detector = self._detector_loader()
            
#         _, aligned_faces = self.detector.detect_and_align(img_bgr)
#         if not aligned_faces:
#             raise ValueError("No face detected.")
#         return self._embed_from_aligned(aligned_faces[0][0])

#     # -------------------- Build DB --------------------
#     def build_embeddings(self, dataset_root: Optional[str] = None, aligned: bool = True, max_images_per_student: int = 5):
#         root = dataset_root or self.dataset_root
#         if not os.path.exists(root):
#             raise FileNotFoundError(f"Dataset not found: {root}")

#         print(f"[BUILD] From dataset: {root}")
#         labels, vecs, failed = [], [], 0

#         for person_dir in tqdm(sorted(os.listdir(root)), desc="Students"):
#             pdir = os.path.join(root, person_dir)
#             if not os.path.isdir(pdir):
#                 continue
#             img_files = [os.path.join(pdir, f) for f in os.listdir(pdir)
#                          if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_images_per_student]
#             emb_list = []
#             for path in img_files:
#                 img = cv2.imread(path)
#                 if img is None:
#                     failed += 1
#                     continue
#                 try:
#                     emb_list.append(self.extract_embedding(img, aligned=aligned))
#                 except Exception as e:
#                     logging.warning(f"Failed to embed {path}: {e}")
#                     failed += 1
#             if emb_list:
#                 m = np.mean(emb_list, axis=0).astype(np.float32)
#                 m /= np.linalg.norm(m)
#                 vecs.append(m)
#                 labels.append(person_dir)

#         if not vecs:
#             print("[ERROR] No valid faces to build DB.")
#             return

#         self.labels = labels
#         self.embeddings = np.vstack(vecs).astype(np.float32)
#         self._ensure_faiss()
#         self._save_db()
#         print(f"[DONE] Built {len(self.labels)} entries (failed images: {failed})")

#     # -------------------- Recognition --------------------
#     def recognize(self, aligned_face: np.ndarray) -> Tuple[str, float]:
#         """Nhận dạng 1 khuôn mặt đã align: trả (label, cosine)."""
#         if len(self.labels) == 0:
#             return "Unknown", 0.0
        
#         # Hàm này giờ đã rất nhanh
#         emb = self._embed_from_aligned(aligned_face)

#         if self.index is not None:
#             q = emb.reshape(1, -1).copy()
#             # faiss.normalize_L2(q) # Không cần vì emb đã được chuẩn hóa
#             sims, ids = self.index.search(q, 1)
#             sim, idx = float(sims[0][0]), int(ids[0][0])
#         else:
#             sims = np.dot(self.embeddings, emb) # Cả 2 vector đều đã được chuẩn hóa
#             idx, sim = int(np.argmax(sims)), float(sims[np.argmax(sims)])

#         return (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)

#     def recognize_faces(self, aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[str, float]]:
#         """Nhận dạng nhiều mặt trong cùng frame (tối ưu theo lô)."""
#         if len(self.labels) == 0 or not aligned_faces:
#             return [("Unknown", 0.0) for _ in aligned_faces]

#         # 1. Trích xuất embedding theo lô (giờ đã rất nhanh)
#         embs = []
#         for img, _ in aligned_faces:
#             try:
#                 embs.append(self._embed_from_aligned(img))
#             except Exception:
#                 embs.append(None) # Thêm None nếu trích xuất thất bại

#         results: List[Tuple[str, float]] = []
#         valid_idx = [i for i, e in enumerate(embs) if e is not None]
#         if not valid_idx: # Nếu không có embedding nào hợp lệ
#             return [("Unknown", 0.0) for _ in aligned_faces]

#         # 2. Tạo ma trận embedding
#         emb_mat = np.vstack([embs[i] for i in valid_idx]).astype(np.float32)
#         # Không cần chuẩn hóa L2 nữa vì _embed_from_aligned đã làm

#         # 3. Tìm kiếm theo lô
#         if self.index is not None:
#             # Q = emb_mat.copy()
#             # faiss.normalize_L2(Q) # Không cần
#             sims, ids = self.index.search(emb_mat, 1)
#             top_sims = sims.ravel().tolist()
#             top_ids = ids.ravel().tolist()
#         else:
#             # [N, 512] @ [512, M] -> [N, M] (N: db, M: query)
#             sims_all = self.embeddings @ emb_mat.T   
#             top_ids = np.argmax(sims_all, axis=0).tolist()
#             top_sims = sims_all[top_ids, range(sims_all.shape[1])].tolist()

#         # 4. Map kết quả
#         res_map = {}
#         for j, i in enumerate(valid_idx):
#             sim, idx = float(top_sims[j]), int(top_ids[j])
#             res_map[i] = (self.labels[idx], sim) if sim >= self.threshold else ("Unknown", sim)
        
#         # Trả về kết quả theo đúng thứ tự ban đầu
#         for i in range(len(aligned_faces)):
#             results.append(res_map.get(i, ("Unknown", 0.0))) # Dùng 0.0 cho các mặt bị lỗi
#         return results

# # =================================================================
# # PHẦN MAIN ĐỂ TEST VÀ VẼ KẾT QUẢ NHẬN DẠNG
# # =================================================================
# def draw_recognition_results(
#     image: np.ndarray,
#     aligned_faces: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
#     results: List[Tuple[str, float]],
# ) -> np.ndarray:
#     """
#     Vẽ bounding box và label (kết quả từ embedding) lên ảnh.
#     """
#     img_draw = image.copy()
#     for i, (label, sim) in enumerate(results):
#         try:
#             # Lấy bbox từ list aligned_faces
#             bbox = aligned_faces[i][1] 
#             x1, y1, x2, y2 = [int(coord) for coord in bbox]

#             # Quyết định màu sắc
#             if label == "Unknown":
#                 color = (0, 0, 255) # Đỏ cho Unknown
#             else:
#                 color = (0, 255, 0) # Xanh lá cho người đã biết

#             # Vẽ bounding box
#             cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

#             # Chuẩn bị text
#             text = f"{label} ({sim:.2f})"
            
#             # Tính kích thước text để vẽ nền
#             (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
#             # Vẽ nền cho text
#             cv2.rectangle(img_draw, (x1, y1 - h - 10), (x1 + w, y1 - 5), color, -1)
#             # Vẽ text
#             cv2.putText(img_draw, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         except Exception as e:
#             logging.warning(f"[Draw] Error drawing bbox {i}: {e}")
            
#     return img_draw


# def main_test():
#     """Hàm main để chạy test nhận dạng trên một ảnh."""
    
#     # --- CẤU HÌNH ---
#     REBUILD_DB = False # Đặt True nếu muốn build lại DB từ dataset
#     # !!! THAY ĐỔI ĐƯỜNG DẪN NÀY
#     TEST_IMAGE_PATH = "/Users/sarahtruc/Documents/System_FaceID/dataset/SinhVien/sontung/004.jpg" 
#     DATASET_ROOT = "dataset/SinhVien"
#     RESULT_SAVE_PATH = "results/demo_recognition.jpg"
    
#     # --- IMPORT ---
#     # Import tại đây để tránh lỗi import vòng nếu file này được import bởi file khác
#     try:
#         from src.face_detect import FaceDetector
#     except ImportError:
#         print("[ERROR] Không thể import FaceDetector.")
#         print("Hãy đảm bảo file 'src/face_detect.py' tồn tại và sys.path đã đúng.")
#         return

#     # --- KHỞI TẠO ---
#     print("[Main] Khởi tạo mô hình...")
    
#     # 1. Dùng chung FaceAnalysis
#     device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#     shared_face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "landmark_3d_68"])
#     shared_face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
#     print("[Main] FaceAnalysis (shared) loaded.")

#     # 2. Detector
#     detector = FaceDetector(
#         device=device,
#         yolo_model_path=os.path.join("models", "yolov11n-face.pt"),
#         face_app=shared_face_app # Dùng chung
#     )
#     print("[Main] FaceDetector loaded.")
    
#     # 3. Recognizer
#     recognizer = FaceRecognizer(
#         device=device,
#         face_app=shared_face_app, # Dùng chung
#         detector=detector, # Cung cấp detector đã load
#         dataset_root=DATASET_ROOT,
#         db_path="encodings/embeddings.pkl"
#     )
#     print("[Main] FaceRecognizer loaded.")

#     # --- BUILD DB (NẾU CẦN) ---
#     if REBUILD_DB:
#         print("[Main] Bắt đầu build lại DB...")
#         # Giả định dataset đã align, nếu chưa, đặt aligned=False
#         recognizer.build_embeddings(aligned=True, max_images_per_student=10)
#         print("[Main] Build DB hoàn tất.")
        
#     if len(recognizer.labels) == 0:
#         print("[ERROR] Database trống. Không thể nhận dạng. Hãy kiểm tra DATASET_ROOT.")
#         return

#     # --- XỬ LÝ ẢNH TEST ---
#     if not os.path.exists(TEST_IMAGE_PATH) or TEST_IMAGE_PATH == "path/to/your/test_image.jpg":
#         print(f"[ERROR] Không tìm thấy ảnh test: {TEST_IMAGE_PATH}")
#         print("Vui lòng cập nhật TEST_IMAGE_PATH trong hàm main_test().")
#         return
        
#     print(f"[Main] Đang xử lý ảnh: {TEST_IMAGE_PATH}")
#     img = cv2.imread(TEST_IMAGE_PATH)
#     if img is None:
#         print(f"[ERROR] Không thể đọc ảnh: {TEST_IMAGE_PATH}")
#         return

#     # 1. Phát hiện và Align
#     # Giả định: detector.detect_and_align trả về (bboxes, List[Tuple[aligned_img, bbox]])
#     # Chúng ta chỉ cần giá trị thứ 2
#     try:
#         _, aligned_faces_list = detector.detect_and_align(img)
#     except Exception as e:
#         print(f"[ERROR] Lỗi khi detect/align: {e}")
#         return
        
#     if not aligned_faces_list:
#         print("[Main] Không tìm thấy khuôn mặt nào trong ảnh.")
#         return
    
#     print(f"[Main] Phát hiện {len(aligned_faces_list)} khuôn mặt.")

#     # 2. Nhận dạng (sử dụng embedding)
#     results = recognizer.recognize_faces(aligned_faces_list)
#     print(f"[Main] Kết quả nhận dạng: {results}")

#     # 3. Vẽ kết quả lên ảnh
#     img_result = draw_recognition_results(img, aligned_faces_list, results)

#     # 4. Lưu và Hiển thị
#     os.makedirs(os.path.dirname(RESULT_SAVE_PATH), exist_ok=True)
#     cv2.imwrite(RESULT_SAVE_PATH, img_result)
#     print(f"[Main] Đã lưu kết quả vào: {RESULT_SAVE_PATH}")
    

#     # Hiển thị (tùy chọn)
#     try:
#         cv2.imshow("Face Recognition Result", img_result)
#         print("Nhấn phím bất kỳ để thoát...")
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except cv2.error:
#         print("[Main] Không thể hiển thị ảnh (có thể do chạy trên server không có GUI).")
#         print("[Main] Vui lòng kiểm tra file kết quả đã lưu.")

# if __name__ == "__main__":
#     main_test()
    
# add_impostor.py
with open("logs/attendance_log.csv", "a", encoding="utf-8") as f:
    for i in range(500):
        f.write("2025-12-03 23:59:59,Impostor,Unknown,0.0000,18.00\n")
print("Đã thêm 500 Impostor (Unknown) thành công!")