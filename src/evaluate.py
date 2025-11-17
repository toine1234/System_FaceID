import os
import numpy as np
import csv
import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ============================================
# CONFIG
# ============================================
DATASET_KNOWN = "dataset/SinhVien"
DATASET_UNKNOWN = "dataset/Unknown"
THRESHOLD = 0.60
DEVICE = 0   # 0: GPU, -1: CPU
LOG_FILE = "logs/attendance_log.csv"

# ============================================
# Init ArcFace
# ============================================
print("[INIT] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=DEVICE, det_size=(320, 320))
print("[READY] Model loaded ✓\n")

# ============================================
# Embedding extractor
# ============================================
def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    emb = faces[0].embedding.astype(np.float32)
    return emb / (norm(emb) + 1e-8)

# ============================================
# Build reference embedding dictionary
# ============================================
def load_reference_embeddings(root):
    database = {}
    for person in os.listdir(root):
        person_path = os.path.join(root, person)
        if not os.path.isdir(person_path):
            continue
        
        embs = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            emb = get_embedding(img_path)
            if emb is not None:
                embs.append(emb)
        
        if embs:
            mean_emb = np.mean(embs, axis=0)
            mean_emb /= (norm(mean_emb) + 1e-8)
            database[person] = mean_emb
    return database

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

# ============================================
# MAIN EVALUATION
# ============================================
def evaluate_system():
    print("[DATA] Loading known embeddings...")
    ref_db = load_reference_embeddings(DATASET_KNOWN)
    print(f"Loaded {len(ref_db)} known identities.\n")

    if len(ref_db) == 0:
        print("[ERROR] No reference embeddings found!")
        return

    # Global counters
    TP = FP = FN = TN = 0

    # Time window evaluation (5 sec)
    window_results = []  # list of dict per evaluation window
    start_time = datetime.now()

    # Evaluate known identities
    for person in ref_db.keys():
        test_dir = os.path.join(DATASET_KNOWN, person)
        for img_name in os.listdir(test_dir):
            img_path = os.path.join(test_dir, img_name)
            emb = get_embedding(img_path)
            if emb is None:
                continue
            
            # cosine match
            sims = {name: cosine_similarity(emb, ref_db[name]) for name in ref_db}
            best_label = max(sims, key=sims.get)
            best_sim = sims[best_label]

            if best_sim >= THRESHOLD:
                if best_label == person:
                    TP += 1
                else:
                    FN += 1
            else:
                FN += 1

            # Evaluate unknown images every loop
            if os.path.exists(DATASET_UNKNOWN):
                for img_name in os.listdir(DATASET_UNKNOWN):
                    img_path = os.path.join(DATASET_UNKNOWN, img_name)
                    emb = get_embedding(img_path)
                    if emb is None:
                        continue
                    
                    sims = {name: cosine_similarity(emb, ref_db[name]) for name in ref_db}
                    best_label = max(sims, key=sims.get)
                    best_sim = sims[best_label]

                    if best_sim >= THRESHOLD:
                        FP += 1
                    else:
                        TN += 1

            # calculate metrics every 5 seconds
            now = datetime.now()
            if (now - start_time).total_seconds() >= 5:
                ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                FAR = FP / (FP + TN + 1e-8)
                FRR = FN / (TP + FN + 1e-8)

                window_results.append({
                    "time": now.strftime("%H:%M:%S"),
                    "ACC": ACC,
                    "FAR": FAR,
                    "FRR": FRR,
                    "TP": TP, "FP": FP, "FN": FN, "TN": TN
                })
                
                start_time = now  # reset window timer

    # Final global metrics
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    FAR = FP / (FP + TN + 1e-8)
    FRR = FN / (TP + FN + 1e-8)

    print("\n=========== FINAL FACEID EVALUATION ===========")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
    print(f"Accuracy = {ACC*100:.2f}%")
    print(f"FAR      = {FAR*100:.2f}%")
    print(f"FRR      = {FRR*100:.2f}%")
    print("===============================================\n")

    # Save GLOBAL CSV
    os.makedirs("logs", exist_ok=True)
    with open("logs/evaluation_result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["TP", TP])
        w.writerow(["FP", FP])
        w.writerow(["FN", FN])
        w.writerow(["TN", TN])
        w.writerow(["Accuracy (%)", f"{ACC*100:.2f}"])
        w.writerow(["FAR (%)", f"{FAR*100:.2f}"])
        w.writerow(["FRR (%)", f"{FRR*100:.2f}"])

    # Save 5-second window data
    df = pd.DataFrame(window_results)
    df.to_csv("logs/Evaluation.csv", index=False)
    print("[SAVED] Evaluation.csv created ✓")

    # Plot chart
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["ACC"], marker='o', label="Accuracy")
    plt.plot(df["time"], df["FAR"], marker='o', label="FAR")
    plt.plot(df["time"], df["FRR"], marker='o', label="FRR")
    plt.grid()
    plt.xlabel("Time Window (5s)")
    plt.ylabel("Metric Value")
    plt.title("FACEID METRICS TREND (5-SECOND WINDOW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("logs/5s_realtime_trend.png", dpi=300)
    plt.show()
    print("[SAVED] logs/5s_realtime_trend.png ✓")

if __name__ == "__main__":
    evaluate_system()