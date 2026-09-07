"""
PatchCamelyon (PCam) dataset loading and preprocessing.

PCam ships as .h5 files (x: images, y: labels). Download from:
https://github.com/basveeling/pcam

Expected files in data/:
    camelyonpatch_level_2_split_train_x.h5
    camelyonpatch_level_2_split_train_y.h5
    camelyonpatch_level_2_split_valid_x.h5
    camelyonpatch_level_2_split_valid_y.h5
    camelyonpatch_level_2_split_test_x.h5
    camelyonpatch_level_2_split_test_y.h5
"""

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PCamDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None, subset_size=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

        with h5py.File(self.x_path, "r") as f:
            self.length = f["x"].shape[0]
        if subset_size is not None:
            self.length = min(self.length, subset_size)

        self._x = None
        self._y = None

    def _lazy_open(self):
        # Open h5 files lazily per-worker to avoid multiprocessing issues.
        if self._x is None:
            self._x = h5py.File(self.x_path, "r")["x"]
            self._y = h5py.File(self.y_path, "r")["y"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._lazy_open()
        image = self._x[idx]
        label = int(self._y[idx].reshape(-1)[0])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


if __name__ == "__main__":
    # Quick sanity check once data is downloaded
    ds = PCamDataset(
        "data/camelyonpatch_level_2_split_train_x.h5",
        "data/camelyonpatch_level_2_split_train_y.h5",
        transform=get_transforms(train=False),
        subset_size=1000,
    )
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Sample image shape: {img.shape}, label: {label}")
