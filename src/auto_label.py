# Phát hiện khuôn mặt bằng YOLO pretrained
# Sinh nhãn .txt

from ultralytics import YOLO
import os
import shutil
import tempfile

def auto_label(image_dir, label_dir, model_path="yolov8n.pt", conf_thres=0.5, output_name="auto_label"):
    project_dir = "runs/"
    model = YOLO(model_path)

    print(f"Auto labeling images in {image_dir}...")

    model.predict(
        source = image_dir,
        conf = conf_thres,
        save_txt = True,
        project = project_dir,
        name = output_name
    )

    gen_labels_dir = os.path.join(project_dir, output_name, "labels")

    if not os.path.exists(gen_labels_dir):
        print(f"[WARN] Labels not found in {gen_labels_dir}")
        return
    
    os.makedirs(label_dir, exist_ok=True)

    for txt_file in os.listdir(gen_labels_dir):
        src_path = os.path.join(gen_labels_dir, txt_file)
        dst_path = os.path.join(label_dir, txt_file)
        with open(src_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = "0"  # ép class thành 0 (face)
                new_lines.append(" ".join(parts))

        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines)) 
        print(f"[OK] {txt_file} -> {label_dir}")

    print(f"[DONE] Auto labeling for {image_dir} completed. Labels are saved in {label_dir}")

if __name__ == "__main__":
    auto_label(
        image_dir = "dataset/images/train",
        label_dir = "dataset/labels/train",
    )

    auto_label(
        image_dir = "dataset/images/val",
        label_dir = "dataset/labels/val",
    )