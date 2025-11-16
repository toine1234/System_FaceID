import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from insightface.app import FaceAnalysis
import cv2
import csv

# ==========================================================
# TP/FP/FN/TN/Accuracy/FAR/FRR as before

# ==========================================================
# 1. Init ArcFace (unify det_size=320)
# ==========================================================
print("[INIT] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(320, 320))  # SỬA: Unify với runtime
print("[READY] ✅ Model loaded\n")

# ==========================================================
# 2. Utilities (normalize emb, mean per person)
# ==========================================================
def get_embedding(img_path):
    """Extract embedding (normalize L2)."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    emb = faces[0].embedding.astype(np.float32)
    return emb / np.linalg.norm(emb)  # SỬA: Normalize L2

def load_dataset(dataset_path, mean_per_person=True):
    """Load embeddings; optional mean per person."""
    person_embs = {}  # Dict: person -> list embs
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        embs = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            emb = get_embedding(img_path)
            if emb is not None:
                embs.append(emb)
        if embs:
            if mean_per_person:
                mean_emb = np.mean(embs, axis=0)
                mean_emb /= np.linalg.norm(mean_emb)  # Re-norm mean
                person_embs[person] = mean_emb
            else:
                person_embs[person] = embs  # All embs if not mean
    X = np.vstack(list(person_embs.values()))
    y = list(person_embs.keys())
    return np.array(X), np.array(y)

# ==========================================================
# 3. Main: Train + Eval
# ==========================================================
def main():
    DATASET = "dataset/SinhVien"
    UNKNOWN = "dataset/Unknown"

    print("[DATA] Loading dataset...")
    X, y = load_dataset(DATASET, mean_per_person=True)  # SỬA: Mean per person
    print(f"Loaded {len(X)} mean embeddings from {len(np.unique(y))} classes")

    # Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save (unchanged)
    os.makedirs("encodings", exist_ok=True)
    pickle.dump((X_train, y_train), open("encodings/embeddings_train.pkl", "wb"))
    pickle.dump((X_test, y_test), open("encodings/embeddings_test.pkl", "wb"))

    # ==========================================================
    # 4. Train SVM (RBF + Scale)
    # ==========================================================
    print("\n[TRAIN] Training SVM classifier...")
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))  # SỬA: RBF + Scale
    clf.fit(X_train, y_train)
    pickle.dump(clf, open("encodings/classifier_svm.pkl", "wb"))
    print("[DONE] ✅ Classifier trained successfully!\n")

    # ==========================================================
    # 5. Evaluate
    # ==========================================================
    print("[EVAL] Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Test Unknown (SỬA: Assume flat dir, all y="Unknown"; predict all as known → FP if not "Unknown")
    X_unknown = []  # Flat load
    unknown_dir = UNKNOWN
    if os.path.exists(unknown_dir):
        for img_name in os.listdir(unknown_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(unknown_dir, img_name)
                emb = get_embedding(img_path)
                if emb is not None:
                    X_unknown.append(emb)
    X_unknown = np.array(X_unknown)
    if len(X_unknown) > 0:
        y_unknown_pred = clf.predict(X_unknown)
        fp = np.sum(y_unknown_pred != "Unknown")  # SỬA: All predicted != "Unknown" → FP
        tn = len(X_unknown) - fp
    else:
        fp, tn = 0, 0

    # Known (unchanged)
    tp = np.sum(y_pred == y_test)
    fn = len(y_test) - tp  # SỬA: Clearer

    FAR = fp / (fp + tn + 1e-8)
    FRR = fn / (fn + tp + 1e-8)

    # ==========================================================
    # 6. Report (unchanged)
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
# 7. Wrapper
# ==========================================================
def evaluate_system():
    main()