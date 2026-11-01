import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def find_classes(base_path):
    """Class names = sub-folders of `base_path`, sorted alphabetically."""
    return sorted(d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)))


class BrainMRIDataset(Dataset):
    """Grayscale MRI slices stored as `base_path/<class>/<image>`.

    Returns tensors of shape (1, img_size, img_size). With `augment=True`,
    Gaussian blur, Gaussian noise and horizontal motion blur are each applied
    with probability 0.5.
    """

    def __init__(self, base_path, classes, augment=False, img_size=224):
        self.image_paths = []
        self.labels = []
        self.augment = augment
        self.classes = classes
        self.img_size = img_size

        for idx, classe in enumerate(classes):
            dossier = os.path.join(base_path, classe)
            if not os.path.isdir(dossier):
                continue
            images = sorted(f for f in os.listdir(dossier) if f.lower().endswith(IMG_EXTENSIONS))
            for img_name in images:
                self.image_paths.append(os.path.join(dossier, img_name))
                self.labels.append(idx)

    def augment_image(self, img):
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = img + noise
        if random.random() > 0.5:  # Motion blur
            size = random.choice([3, 5, 7])
            kernel = np.zeros((size, size), dtype=np.float32)
            kernel[(size - 1) // 2, :] = 1.0 / size
            img = cv2.filter2D(img, -1, kernel)
        return img

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if img.shape != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img = self.augment_image(img)

        # Normalization
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)
        mask = img > 0
        if mask.any():
            mean = np.mean(img[mask])
            std = np.std(img[mask])
            img[mask] = (img[mask] - mean) / (std + 1e-8)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)   # (1, H, W)
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = self.preprocess(img)
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class BrainMRITestDataset(BrainMRIDataset):
    """Version sans augmentation pour les datasets de test (Blurred, Noisy, Motion)"""

    def __init__(self, base_path, classes, img_size=224):
        super().__init__(base_path, classes, augment=False, img_size=img_size)
