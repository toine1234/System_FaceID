import os
import cv2
from ultralytics import YOLO

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Chuyển từ bounding box (pixel) sang định dạng YOLO (normalized).
    Trả về (x_center, y_center, width, height).
    """
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2

    x_center /= img_w
    y_center /= img_h
    box_w /= img_w
    box_h /= img_h

    return x_center, y_center, box_w, box_h

def auto_label_images(model_path, images_dir, labels_dir, class_id=0, conf_thresh=0.5):
    model = YOLO(model_path)
    ensure_dir(labels_dir)

    for img_fname in os.listdir(images_dir):
        if not (img_fname.lower().endswith(".jpg") or img_fname.lower().endswith(".png")):
            continue

        img_path = os.path.join(images_dir, img_fname)
        results = model.predict(source=img_path, conf=conf_thresh, save=False, verbose=False)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        label_fname = os.path.splitext(img_fname)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_fname)

        with open(label_path, "w") as f:
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    xc, yc, bw, bh = convert_box_to_yolo(
                        float(x1), float(y1), float(x2), float(y2), img_w, img_h
                    )
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"[OK] {img_fname} → {label_fname}")

if __name__ == "__main__":
    model_path = "/Users/sarahtruc/Documents/System_FaceID/models/yolov8n-face.pt"

    images_root = "dataset/images"
    labels_root = "dataset/labels"

    for split in ["train", "val"]:
        img_dir = os.path.join(images_root, split)
        lbl_dir = os.path.join(labels_root, split)
        print(f"\n🚀 Auto-labeling {split} set ...")
        auto_label_images(model_path, img_dir, lbl_dir, class_id=0, conf_thresh=0.5)

    print("\n[DONE] Auto-labeling completed with YOLOv8-Face!")
