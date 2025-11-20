import os
import numpy as np
import csv
import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================
# CONFIG
# ============================================
DATASET_KNOWN = "dataset/SinhVien"
DATASET_UNKNOWN = "dataset/Unknown"
THRESHOLD = 0.60
DEVICE = -1       # -1 = CPU, 0 = GPU

OUT_DIR = "logs"
RESULT_CSV = f"{OUT_DIR}/evaluation_result.csv"
PLOT_PATH = f"{OUT_DIR}/evaluation.png"

# ============================================
# INIT ARCface
# ============================================
print("[INIT] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=DEVICE, det_size=(320, 320))
print("[READY] Model loaded ✓\n")

# ============================================
# EMBEDDING
# ============================================
def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    emb = faces[0].embedding.astype(np.float32)
    return emb / (norm(emb) + 1e-8)

def build_reference_embeddings(root):
    db = {}
    for person in os.listdir(root):
        person_dir = os.path.join(root, person)
        if not os.path.isdir(person_dir):
            continue

        features = []
        for img_name in os.listdir(person_dir):
            emb = get_embedding(os.path.join(person_dir, img_name))
            if emb is not None:
                features.append(emb)

        if features:
            mean_emb = np.mean(features, axis=0)
            mean_emb /= (norm(mean_emb) + 1e-8)
            db[person] = mean_emb
    return db

def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

# ============================================
# MAIN EVALUATION
# ============================================
def evaluate_system():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[STEP] Building reference embeddings...")
    ref_db = build_reference_embeddings(DATASET_KNOWN)
    print(f"Loaded {len(ref_db)} identities\n")

    if not ref_db:
        print("[ERROR] No reference DB found!")
        return

    TP = TN = FP = FN = 0

    # ----------------------------------------
    # Evaluate KNOWN → TP & FN
    # ----------------------------------------
    print("[STEP] Evaluating KNOWN samples...")
    for person in ref_db:
        person_dir = os.path.join(DATASET_KNOWN, person)
        for img_name in os.listdir(person_dir):
            emb = get_embedding(os.path.join(person_dir, img_name))
            if emb is None:
                continue

            scores = {name: cos_sim(emb, ref_db[name]) for name in ref_db}
            best_label, best_score = max(scores, key=scores.get), max(scores.values())

            if best_score >= THRESHOLD:
                if best_label == person:
                    TP += 1
                else:
                    FN += 1
            else:
                FN += 1

    # ----------------------------------------
    # Evaluate UNKNOWN → TN & FP
    # ----------------------------------------
    print("[STEP] Evaluating UNKNOWN samples...")
    if os.path.exists(DATASET_UNKNOWN):
        for img_name in os.listdir(DATASET_UNKNOWN):
            emb = get_embedding(os.path.join(DATASET_UNKNOWN, img_name))
            if emb is None:
                continue

            scores = {name: cos_sim(emb, ref_db[name]) for name in ref_db}
            best_score = max(scores.values())

            if best_score >= THRESHOLD:
                FP += 1              # unknown nhưng nhận nhầm → false acceptance
            else:
                TN += 1
    else:
        print("[WARN] Unknown folder missing!")

    # ----------------------------------------
    # Compute Metrics
    # ----------------------------------------
    total = TP + TN + FP + FN

    ACC = (TP + TN) / (total + 1e-8)
    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (TP + FN + 1e-8)

    print("\n=========== FINAL RESULT ===========")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
    print(f"Accuracy = {ACC*100:.2f}%")
    print(f"FAR      = {FAR*100:.2f}%")
    print(f"FRR      = {FRR*100:.2f}%")
    print("====================================\n")

    # Save CSV
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["TP", TP])
        w.writerow(["FP", FP])
        w.writerow(["FN", FN])
        w.writerow(["TN", TN])
        w.writerow(["Accuracy (%)", f"{ACC*100:.2f}"])
        w.writerow(["FAR (%)", f"{FAR*100:.2f}"])
        w.writerow(["FRR (%)", f"{FRR*100:.2f}"])

    print(f"[SAVED] {RESULT_CSV}")

    # ----------------------------------------
    # Draw 1 Beautiful Chart
    # ----------------------------------------
    metrics = ["Accuracy", "FAR", "FRR"]
    values = [ACC*100, FAR*100, FRR*100]

    plt.figure(figsize=(7, 6))
    bars = plt.bar(metrics, values, width=0.55)

    for b, val in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, val + 1,
                 f"{val:.2f}%", ha='center', fontsize=12, fontweight='bold')

    plt.ylabel("Percentage (%)")
    plt.title("FaceID Evaluation Metrics", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    print(f"[SAVED] {PLOT_PATH}")

    print("\n[✓] EVALUATION COMPLETED\n")


if __name__ == "__main__":
    evaluate_system()
