"""
Dataset handling and image preprocessing utilities.
Author: Gogul Gupta (Team Shree Yantra Dynamics)
"""

import os
from typing import Tuple, Optional, Callable, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization statistics used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: Tuple[int, int] = (224, 224), is_train: bool = False) -> transforms.Compose:
    """
    Returns image transformation pipelines for training or inference.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class SoilDataset(Dataset):
    """
    PyTorch Dataset for loading labeled or unlabeled soil images.
    """
    def __init__(
        self,
        image_dir: str,
        df: Optional[pd.DataFrame] = None,
        image_col: str = "image_id",
        label_col: Optional[str] = "label",
        transform: Optional[Callable] = None
    ):
        self.image_dir = image_dir
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform if transform else get_transforms(is_train=False)

        if self.df is None:
            # Load all image filenames directly from the directory
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.image_names = [
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
        else:
            self.image_names = self.df[self.image_col].tolist()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.df is not None and self.label_col and self.label_col in self.df.columns:
            label = self.df.iloc[idx][self.label_col]
            return image, label, img_name

        return image, img_name
