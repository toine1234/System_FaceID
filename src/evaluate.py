import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from insightface.app import FaceAnalysis
import cv2
import csv

# ==========================================================
# 1. Khởi tạo model trích xuất embedding (ArcFace)
# ==========================================================
print("[INIT] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(160, 160))
print("[READY] ✅ Model loaded\n")


# ==========================================================
# 2. Hàm tiện ích
# ==========================================================
def get_embedding(img_path):
    """Trích xuất embedding cho ảnh"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding


def load_dataset(dataset_path):
    """Đọc tất cả ảnh trong dataset và trích embedding"""
    X, y = [], []
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            emb = get_embedding(img_path)
            if emb is not None:
                X.append(emb)
                y.append(person)
    return np.array(X), np.array(y)


# ==========================================================
# 3. Hàm chính: train + evaluate
# ==========================================================
def main():
    DATASET = "dataset/SinhVien"
    UNKNOWN = "dataset/Unknown"

    print("[DATA] Loading dataset...")
    X, y = load_dataset(DATASET)
    print(f"Loaded {len(X)} face embeddings from {len(np.unique(y))} classes")

    # Tách train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Lưu lại embeddings
    os.makedirs("encodings", exist_ok=True)
    pickle.dump((X_train, y_train), open("encodings/embeddings_train.pkl", "wb"))
    pickle.dump((X_test, y_test), open("encodings/embeddings_test.pkl", "wb"))

    # ==========================================================
    # 4. Huấn luyện Classifier (SVM)
    # ==========================================================
    print("\n[TRAIN] Training SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)
    pickle.dump(clf, open("encodings/classifier_svm.pkl", "wb"))
    print("[DONE] ✅ Classifier trained successfully!\n")

    # ==========================================================
    # 5. Đánh giá trên tập test
    # ==========================================================
    print("[EVAL] Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Test Unknown
    X_unknown, y_unknown = load_dataset(UNKNOWN)
    if len(X_unknown) > 0:
        y_unknown_pred = clf.predict(X_unknown)
        fp = np.sum(y_unknown_pred != "Unknown")  # nhận nhầm người lạ
        tn = len(y_unknown) - fp
    else:
        fp, tn = 0, 0

    # Test Known
    tp = np.sum(y_pred == y_test)
    fn = np.sum(y_pred != y_test)

    FAR = fp / (fp + tn + 1e-8)
    FRR = fn / (fn + tp + 1e-8)

    # ==========================================================
    # 6. Xuất báo cáo
    # ==========================================================
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/evaluation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["True Positive (TP)", tp])
        writer.writerow(["False Positive (FP)", fp])
        writer.writerow(["False Negative (FN)", fn])
        writer.writerow(["True Negative (TN)", tn])
        writer.writerow(["Accuracy", f"{acc*100:.2f}%"])
        writer.writerow(["FAR", f"{FAR*100:.2f}%"])
        writer.writerow(["FRR", f"{FRR*100:.2f}%"])

    print("\n================ Evaluation Report ================")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"FAR: {FAR*100:.2f}%")
    print(f"FRR: {FRR*100:.2f}%")
    print("===================================================")
    print(f"[SAVED] Report saved at: {csv_path}")


# ==========================================================
# 7. Hàm cho app.py gọi tự động
# ==========================================================
def evaluate_system():
    """Hàm wrapper để app.py có thể gọi quá trình đánh giá"""
    main()
