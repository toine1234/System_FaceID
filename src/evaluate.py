import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib


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

    # ===== PRECISION & RECALL =====
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0


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

    print(f"Precision    : {Precision*100:.2f}%")
    print(f"Recall       : {Recall*100:.2f}%")

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
        "Precision": round(Precision, 4),
        "Recall": round(Recall, 4),
    }])
    result.to_csv(OUTPUT_SUMMARY, index=False)
    print("[SAVED] Summary →", OUTPUT_SUMMARY)

    # ==========================================
    #   PLOT: Accuracy – Precision – Recall – FAR – FRR
    # ==========================================
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    metrics_all = ["Accuracy", "Precision", "Recall", "FAR", "FRR"]
    values_all = [Accuracy*100, Precision*100, Recall*100, FAR*100, FRR*100]
    colors_all = ["#6BA292", "#5DA5DA", "#9CC3E4", "#F15854", "#F5A623"]
    bars = plt.bar(metrics_all, values_all, 
                   color=colors_all, 
                   alpha=0.92,
                   edgecolor="#333333",
                   linewidth=1.2)
    for bar in bars:
        bar.set_linewidth(0)
        bar.set_zorder(3)

    plt.title("SYSTEM EVALUATION (%)", 
              fontsize=20, 
              fontweight="bold",
              pad=20)

    plt.ylabel("Percentage (%)", fontsize=13)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)

    for bar, value in zip(bars, values_all):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )

    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/system_evaluate.png", dpi=300)
    plt.show()

    # # ===== FPS CHART =====
    # plt.figure(figsize=(12, 5))
    # plt.plot(df.index, df["fps"], "-o", label="FPS", color="purple")
    # plt.title("FPS Over Time")
    # plt.xlabel("Frame Index")
    # plt.ylabel("FPS")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{CHART_DIR}/fps.png", dpi=300)
    # plt.show()

    # ===== METRICS CHART PER FRAME =====
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df["correct"].astype(int), "-o", label="Correct (TP)")
    # plt.plot(df.index, df["false_accept"].astype(int), "-o", label="False Accept (FP)", color="red")
    # plt.plot(df.index, df["false_reject"].astype(int), "-o", label="False Reject (FN)", color="orange")

    # plt.title("TP / FP / FN per frame")
    # plt.xlabel("Frame Index")
    # plt.ylabel("Value (0 or 1)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{CHART_DIR}/metrics.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    evaluate_system()
