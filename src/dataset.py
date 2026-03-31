import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(self, base_path, classes, augment=False):
        self.image_paths = []
        self.labels = []
        self.augment = augment
        self.classes = classes

        for idx, classe in enumerate(classes):
            dossier = os.path.join(base_path, classe)
            if not os.path.isdir(dossier):
                continue
            images = [f for f in os.listdir(dossier) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in images:
                self.image_paths.append(os.path.join(dossier, img_name))
                self.labels.append(idx)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            if random.random() > 0.5:
                noise = np.random.normal(0, 5, img.shape)
                img = img + noise
            if random.random() > 0.5:  # Motion blur
                size = random.choice([3, 5, 7])
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                img = cv2.filter2D(img, -1, kernel)

        # Normalization
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)
        mask = img > 0
        mean = np.mean(img[mask])
        std = np.std(img[mask])
        img[mask] = (img[mask] - mean) / (std + 1e-8)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)   # (1, H, W) for SimpleCNN
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = self.preprocess(img)
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class BrainMRITestDataset(Dataset):
    """Version sans augmentation pour les datasets de test (Blurred, Noisy, Motion)"""
    def __init__(self, base_path, classes):
        self.image_paths = []
        self.labels = []
        self.classes = classes

        for idx, classe in enumerate(classes):
            dossier = os.path.join(base_path, classe)
            if not os.path.isdir(dossier):
                continue
            images = [f for f in os.listdir(dossier) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in images:
                self.image_paths.append(os.path.join(dossier, img_name))
                self.labels.append(idx)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)
        mask = img > 0
        mean = np.mean(img[mask])
        std = np.std(img[mask])
        img[mask] = (img[mask] - mean) / (std + 1e-8)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = self.preprocess(img)
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
