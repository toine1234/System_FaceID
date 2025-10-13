import os
import cv2

def convert_to_yolo_line(x, y, w, h, img_w, img_h):
    """
    Convert từ bounding box (pixel) sang YOLO format (normalized).
    YOLO format: class_id x_center y_center width height
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

def parse_annotations(ann_file, img_dir, out_dir):
    """
    Convert annotation WIDER-FACE sang YOLO format.
    - ann_file: file annotation (.txt)
    - img_dir: thư mục ảnh gốc (WIDER_train/images hoặc WIDER_val/images)
    - out_dir: thư mục output (labels/train hoặc labels/val)
    """
    with open(ann_file, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        # Nếu dòng chứa ảnh
        if ".jpg" in line:
            img_rel_path = line
            idx += 1

            # số khuôn mặt trong ảnh
            face_count = int(lines[idx].strip())
            idx += 1

            img_path = os.path.join(img_dir, img_rel_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: Could not read image: {img_path}")
                idx += face_count
                continue
            h, w = img.shape[:2]

            label_lines = []
            for _ in range(face_count):
                parts = lines[idx].strip().split()
                x, y, bw, bh = map(int, parts[:4])
                idx += 1
                if bw <= 0 or bh <= 0:
                    continue
                label_lines.append(convert_to_yolo_line(x, y, bw, bh, w, h))

            # Lưu file nhãn YOLO
            out_path = os.path.join(out_dir, os.path.splitext(img_rel_path)[0] + ".txt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f_out:
                f_out.writelines(label_lines)
        else:
            idx += 1

if __name__ == "__main__":
    base = "C:/Users/DELL/Desktop/System_FaceID/dataset_WIDER-FACE"

    # Convert train
    parse_annotations(
        os.path.join(base, "wider_face_split/wider_face_train_bbx_gt.txt"),
        os.path.join(base, "WIDER_train/images"),
        "C:/Users/DELL/Desktop/System_FaceID/dataset/labels/train"
    )

    # Convert val
    parse_annotations(
        os.path.join(base, "wider_face_split/wider_face_val_bbx_gt.txt"),
        os.path.join(base, "WIDER_val/images"),
        "C:/Users/DELL/Desktop/System_FaceID/dataset/labels/val"
    )
    print("✅ Annotation conversion to YOLO format complete.")
