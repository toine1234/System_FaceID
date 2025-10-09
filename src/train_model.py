# Huấn luyện mô hình nhận dạng

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
    return yaml_file

def train_yolo(yaml_file, model_name="yolov8n.pt", epochs=20, imgsz=640, batch=8, lr0=0.001, lrf=0.001, optimizer="AdamW", project_name="yolov8-face"):
    model = YOLO(model_name)

    for param in model.model.parameters():
        param.requires_grad = False

    last_layer = list(model.model.children())[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    # Train
    model.train(
        data=yaml_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        name=project_name
    )
    return model

if __name__ == "__main__":
    train_path = "dataset/images/train"
    val_path   = "dataset/images/val"

    # 1. Tạo file data.yaml
    yaml_file = create_data_yaml(train_path, val_path)

    # 2. Train YOLO
    train_yolo(yaml_file, epochs=20, batch=8, imgsz=640)