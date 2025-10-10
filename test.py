from ultralytics import YOLO

model = YOLO("/Users/sarahtruc/Documents/System_FaceID/models/yolov8n-face.pt")
print("Task:", model.task)   # detect / pose / segment
print("Names:", model.names) # class names
