System FaceID
# FaceID Attendance System

Hệ thống quản lý học tập tích hợp tính năng nhận diện khuôn mặt (FaceID) sử dụng **OpenCV**, **YOLOv8-face**, và **CNN**.  
Ngôn ngữ : Python
Mục tiêu: Tự động điểm danh, quản lý sinh viên và tích hợp với hệ thống quanr lý học tập H2T

## Cấu trúc dự án

FaceID-System/
│
├── dataset/                # Dữ liệu gốc + nhãn YOLO
│   ├── images/              # chứa ảnh gốc (train/val/test)
│   │   ├── train/           # ảnh dùng để huấn luyện
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   ├── val/   # ảnh dùng để kiểm thử trong lúc train (validation set)
│   │   │   ├── imgA.jpg
│   │   │   ├── imgB.jpg
│   │   │   └── ...
    ├── labels/ # chứa nhãn (annotations) tương ứng cho từng ảnh
    │   ├── train/   # file nhãn của ảnh trong images/train
    │   │   ├── img1.txt
    │   │   ├── img2.txt
│   │   │   └── ...
│   │   ├── val/     # file nhãn của ảnh trong images/val
│   │   │   ├── imgA.txt
│   │   │   ├── imgB.txt
│   │   │   └── ...
│
│
├── models/                 # Kết quả training (best, result)
│   ├── face_detector/      # YOLOv8/MTCNN/RetinaFace (dùng để detect)
│   └── face_recognizer/    # ArcFace/FaceNet đã huấn luyện
│
├── encodings/              # Vector embedding (128D/512D)
│   ├── embeddings.pkl      # File chứa embedding và nhãn sinh viên
│
├── src/                    # Code nguồn chính
│   ├── data_loader.py      # Quản lý dataset, sinh batch huấn luyện
│   ├── face_detect.py      # Phát hiện khuôn mặt (OpenCV/YOLO)
│   ├── face_recognize.py   # Sinh embedding & nhận dạng
│   ├── train_model.py      # Huấn luyện mô hình nhận dạng (ArcFace/FaceNet)
│   ├── evaluate.py         # Kiểm thử độ chính xác (accuracy, FAR/FRR, ROC)
│   ├── utils.py            # Hàm tiện ích (resize, augment, logging)
│   └── camera_demo.py      # Demo nhận diện real-time từ webcam
│
├── api/                    # Chuẩn bị cho tích hợp sau này
│   ├── app.py              # Flask/FastAPI server cung cấp REST API
│   └── routes.py           # Các route /detect /recognize
│
├── tests/                  # Unit test cho từng module
│   ├── test_detection.py
│   ├── test_recognition.py
│
├── requirements.txt        # Danh sách thư viện (opencv, torch, ultralytics,…)
├── README.md               # Hướng dẫn sử dụng
└── config.yaml             # Cấu hình (đường dẫn dataset, tham số huấn luyện)
