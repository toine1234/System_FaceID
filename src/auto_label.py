import os
import cv2
from ultralytics import YOLO

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Chuyển từ bounding box pixel sang định dạng YOLO (x_center, y_center, width, height)
    """
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2
    return x_center/img_w, y_center/img_h, box_w/img_w, box_h/img_h

def auto_label_dataset(model_path, images_root, labels_root, conf_thresh=0.5):
    """
    Dán nhãn tự động cho dataset sinh viên.
    Mỗi thư mục con trong images_root là tên sinh viên.
    """
    model = YOLO(model_path)

    for student_name in os.listdir(images_root):
        student_img_dir = os.path.join(images_root, student_name)
        if not os.path.isdir(student_img_dir):
            continue

        # Thư mục label tương ứng
        student_label_dir = os.path.join(labels_root, student_name)
        ensure_dir(student_label_dir)

        for img_file in os.listdir(student_img_dir):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(student_img_dir, img_file)
            label_path = os.path.join(student_label_dir, os.path.splitext(img_file)[0] + ".txt")

            results = model(img_path)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]

            with open(label_path, "w") as f:
                for box in results[0].boxes:
                    conf = float(box.conf)
                    if conf >= conf_thresh:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_c, y_c, w, h = convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h)
                        f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            print(f"✅ Labeled {student_name}/{img_file}")

if __name__ == "__main__":
    model_path = "models/face_detector/yolov8n-face/train_results/weights/best.pt"
    images_root = "dataset/images"
    labels_root = "dataset/labels"
    auto_label_dataset(model_path, images_root, labels_root)
