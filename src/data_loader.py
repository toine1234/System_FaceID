# File quản lý dataset, chia batch, và transform ảnh.

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceDetection(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform

        # get list of image files and annotation files
        self.image_paths = sorted([
            os.path.join(image_folder, fname) 
            for fname in os.listdir(image_folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.annotation_paths = sorted([
            os.path.join(annotation_folder, fname) 
            for fname in os.listdir(annotation_folder)
            if fname.lower().endswith('.txt')
        ])

        assert len(self.image_paths) == len(self.annotation_paths), \
            f"Number of images ({len(self.image_paths)}) and annotation files ({len(self.annotation_paths)}) do not match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]

        img = Image.open(img_path).convert('RGB')

        boxes = []
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            boxes.append([class_id, x_center, y_center, w, h])

        boxes = np.array(boxes, dtype=np.float32)  # shape (N, 5)

        if self.transform:
            img = self.transform(img)

        return img, boxes
    
def get_dataloader(image_folder, annotation_folder, batch_size=8, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = FaceDetection(image_folder, annotation_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    return dataloader