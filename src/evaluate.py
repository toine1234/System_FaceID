import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

LOG_FILE = "logs/attendance_log.csv"
OUTPUT_SUMMARY = "logs/evaluate_result.csv"
CHART_DIR = "logs/charts"


def load_log():
    if not os.path.exists(LOG_FILE):
        print("[ERROR] attendance_log.csv NOT FOUND!")
        return None

    df = pd.read_csv(LOG_FILE, header=None)
    df.columns = ["timestamp", "expected_id", "predicted_id", "confidence", "fps"]

    df["expected_id"] = df["expected_id"].astype(str)
    df["predicted_id"] = df["predicted_id"].astype(str)

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["fps"] = pd.to_numeric(df["fps"], errors="coerce")

    return df


def evaluate_system():
    df = load_log()
    if df is None:
        return

    # ===== AUTO-DETECT EXPECTED ID =====
    expected_id = df["expected_id"].mode()[0]
    print(f"[INFO] Auto-detected expected_id = {expected_id}")

    # ===== TRUE / FALSE =====
    df["correct"] = df["predicted_id"] == expected_id
    df["false_accept"] = (df["predicted_id"] != expected_id) & (df["predicted_id"] != "Unknown")
    df["false_reject"] = df["predicted_id"] == "Unknown"

    # ===== CONFUSION VALUES =====
    TP = df["correct"].sum()
    FP = df["false_accept"].sum()
    FN = df["false_reject"].sum()
    TN = len(df) - TP - FP - FN
    total = len(df)

    # ===== METRICS =====
    Accuracy = (TP + TN) / total
    FAR = FP / total
    FRR = FN / total

    avg_conf = df["confidence"].mean()
    avg_fps = df["fps"].mean()

    # ===== PRINT RESULT =====
    print("\n========== FACEID EVALUATION SUMMARY ==========")
    print(f"Expected ID  : {expected_id}")
    print(f"Total Frames : {total}")
    print(f"TP           : {TP}")
    print(f"TN           : {TN}")
    print(f"FP (FAR)     : {FP}")
    print(f"FN (FRR)     : {FN}\n")

    print(f"Accuracy     : {Accuracy*100:.2f}%")
    print(f"FAR          : {FAR*100:.2f}%")
    print(f"FRR          : {FRR*100:.2f}%")
    print(f"Avg Confidence: {avg_conf:.4f}")
    print(f"Avg FPS      : {avg_fps:.2f}")
    print("================================================\n")

    # ===== SAVE CSV =====
    result = pd.DataFrame([{
        "Expected ID": expected_id,
        "Total Frames": total,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Accuracy": round(Accuracy, 4),
        "FAR": round(FAR, 4),
        "FRR": round(FRR, 4),
        "Avg Confidence": round(avg_conf, 4),
        "Avg FPS": round(avg_fps, 2),
    }])
    result.to_csv(OUTPUT_SUMMARY, index=False)
    print("[SAVED] Summary →", OUTPUT_SUMMARY)

    # ==========================================
    #       PLOT ACCURACY – FAR – FRR
    # ==========================================
    os.makedirs(CHART_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))

    metrics = ["Accuracy", "FAR", "FRR"]
    values = [Accuracy, FAR, FRR]
    colors = ["green", "red", "orange"]

    plt.bar(metrics, values, color=colors, alpha=0.85)
    plt.title("SYSTEM EVALUATION: Accuracy – FAR – FRR")
    plt.ylabel("Value (0 to 1)")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(values):
        plt.text(i, v + 0.03, f"{v*100:.2f}%", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/system_evaluation.png", dpi=300)
    plt.show()

    # ===== METRICS CHART PER FRAME =====
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["correct"].astype(int), "-o", label="Correct (TP)")
    plt.plot(df.index, df["false_accept"].astype(int), "-o", label="False Accept (FP)", color="red")
    plt.plot(df.index, df["false_reject"].astype(int), "-o", label="False Reject (FN)", color="orange")

    plt.title("TP / FP / FN per frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Value (0 or 1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/metrics.png", dpi=300)
    plt.show()

    # ===== FPS CHART =====
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["fps"], "-o", label="FPS", color="purple")
    plt.title("FPS Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("FPS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/fps.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    evaluate_system()
