# Huấn luyện mô hình nhận dạng

import os
import yaml
from ultralytics import YOLO

def create_data_yaml(train_path, val_path, yaml_file = "data.yaml"):
    data = {
        'train': train_path,
        'val': val_path,
        'nc': 1,
        'names': ['face']
    }
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f)
    print(f"✅ Data config saved to: {yaml_file}")
    return yaml_file

def train_yolo(yaml_file, model_name="models/yolov8n-face.pt", epochs=20, imgsz=640, batch=8, lr0=0.001, lrf=0.001, optimizer="AdamW", project_name="models/face_detector/yolov8n-face"):
    model = YOLO(model_name)

    # Train
    model.train(
        data=yaml_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        project=project_name ,
        name="train_results",
        device='cpu'
    )
    print(f"🎯 Training complete! Results saved in: {project_name}/train_results")
    return model

if __name__ == "__main__":
    train_path = "dataset/images/train"
    val_path   = "dataset/images/val"

    yaml_file = create_data_yaml(train_path, val_path)

    train_yolo(yaml_file, epochs=20, batch=8, imgsz=640)