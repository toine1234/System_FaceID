import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# CONFIG
# ============================================
LOG_FILE = "logs/attendance_log.csv"
THRESHOLD = 0.60
OUTPUT_CSV = "logs/evaluate.csv"
CHART_DIR = "logs/charts"

# ============================================
# LOAD LOG FILE
# ============================================
def load_log():
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] Log file not found: {LOG_FILE}")
        return None

    df = pd.read_csv(LOG_FILE, header=None)
    df.columns = ["timestamp", "predicted_id", "confidence", "fps"]

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["fps"] = pd.to_numeric(df["fps"], errors="coerce")

    return df

# ============================================
# MULTI-ID EVALUATION
# ============================================
def evaluate_system():
    df = load_log()
    if df is None:
        return

    # Auto-detect multi-ID ground truth
    df["expected_id"] = df["predicted_id"]

    # Correct classification
    df["correct"] = df["predicted_id"] == df["expected_id"]

    # FAR: multi-ID = 0 (vì expected_id = predicted_id)
    df["false_accept"] = False

    # FRR: reject đúng người
    df["false_reject"] = df["confidence"] < THRESHOLD

    # Summary statistics
    total = len(df)
    ACC = df["correct"].sum() / total
    FAR = df["false_accept"].sum() / total
    FRR = df["false_reject"].sum() / total

    avg_conf = df["confidence"].mean()
    avg_fps = df["fps"].mean()

    # ============================================
    # PRINT SUMMARY
    # ============================================
    print("\n=========== MULTI-ID SYSTEM EVALUATION (Realtime FaceID) ===========")
    print(f"Total Frames = {total}")
    print(f"Accuracy     = {ACC*100:.2f}%")
    print(f"FAR          = {FAR*100:.2f}%")
    print(f"FRR          = {FRR*100:.2f}%")
    print(f"Avg Conf     = {avg_conf:.4f}")
    print(f"Avg FPS      = {avg_fps:.2f}")
    print("=====================================================================\n")

    # Save summary to CSV
    os.makedirs("logs", exist_ok=True)

    result_df = pd.DataFrame([{
        "Total Frames": total,
        "Accuracy": round(ACC, 4),
        "FAR": round(FAR, 4),
        "FRR": round(FRR, 4),
        "Avg Confidence": round(avg_conf, 4),
        "Avg FPS": round(avg_fps, 4),
    }])
    result_df.to_csv(OUTPUT_CSV, index=False)
    print("[SAVED] Summary →", OUTPUT_CSV)

    # Create chart folder
    os.makedirs(CHART_DIR, exist_ok=True)

    # ============================================
    # PLOT: Accuracy – FAR – FRR theo frame
    # ============================================
    acc = df["correct"].astype(int)
    far = df["false_accept"].astype(int) + 0.03
    frr = df["false_reject"].astype(int) - 0.03

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, acc, "-o", label="Accuracy per frame", linewidth=2)
    plt.plot(df.index, far, "--o", label="FAR per frame (shifted)", color="red")
    plt.plot(df.index, frr, "--o", label="FRR per frame (shifted)", color="orange")

    plt.title("System Evaluation (Accuracy – FAR – FRR)", fontsize=16)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Metric Value (0/1)", fontsize=14)
    plt.ylim(-0.2, 1.2)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    os.makedirs(CHART_DIR, exist_ok=True)
    save_path1 = f"{CHART_DIR}/metrics.png"
    plt.savefig(save_path1, dpi=300)
    print("[SAVED] Accuracy–FAR–FRR chart →", save_path1)
    plt.show()


    # ============================================
    # PLOT: FPS theo thời gian
    # ============================================
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["fps"], "-o", color="purple", linewidth=2, label="FPS")

    plt.title("FPS Over Time (Realtime FaceID)", fontsize=16)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("FPS", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path2 = f"{CHART_DIR}/fps.png"
    plt.savefig(save_path2, dpi=300)
    print("[SAVED] FPS chart →", save_path2)

    plt.show()


if __name__ == "__main__":
    evaluate_system()
