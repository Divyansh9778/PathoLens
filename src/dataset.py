"""
PatchCamelyon (PCam) dataset loading and preprocessing.

Uses torchvision's built-in PCAM dataset class, which handles downloading,
extracting, and reading the underlying HDF5 files internally — no manual
h5py file handling needed.

Splits (torchvision's naming, not the original PCam repo's):
    "train" - 262,144 patches, ~6.8GB (not used in this project - see README
               scope decisions; too large for a 10-day timeline)
    "val"   - 32,768 patches, ~800MB (used here as our *training* set)
    "test"  - 32,768 patches, ~800MB (used here as our *validation* set)

Yes, this means our "train" data is technically PCam's "val" split, and our
"val" data is PCam's "test" split. That's a deliberate, documented tradeoff
(see README) — not a bug.
"""

import torch
from torchvision import transforms
from torchvision.datasets import PCAM


def get_pcam_dataset(data_dir, split, train, subset_size=None, download=True):
    """
    data_dir: root folder passed to torchvision (it manages its own
              'pcam/' subfolder inside this)
    split: one of "train", "val", "test" (torchvision's naming)
    train: whether to apply training augmentations or eval-only transforms
    subset_size: if set, wraps the dataset in a Subset of this many samples
    """
    dataset = PCAM(
        root=data_dir,
        split=split,
        transform=get_transforms(train=train),
        download=download,
    )

    if subset_size is not None and subset_size < len(dataset):
        indices = list(range(subset_size))
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset


def get_transforms(train=True):
    # No ToPILImage() here - PCAM already yields PIL Images directly,
    # unlike the raw numpy arrays the old manual h5py loader returned.
    if train:
        # Stronger augmentation than the original mild flips + slight jitter,
        # added to combat measured overfitting. Histopathology patches have no
        # canonical orientation, so full 90-degree rotations and both flips are
        # label-preserving and effectively multiply the training set. Heavier
        # colour jitter (plus slight hue shift) simulates the real stain
        # variation between labs/scanners, which is a well-known source of
        # distribution shift in digital pathology.
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                    saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", default="test",
                         help="train/val/test - use 'test' or 'val' for a "
                              "quick check, 'train' will trigger a ~6.8GB download")
    args = parser.parse_args()

    ds = get_pcam_dataset(args.data_dir, split=args.split, train=False, subset_size=1000)
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Sample image shape: {img.shape}, label: {label}")