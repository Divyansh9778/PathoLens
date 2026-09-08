"""
Build a synthetic "WSI-scale" mosaic image by stitching together real
PatchCamelyon patches into a large grid.

This is a deliberate, documented scope decision (see README) — it lets the
whole tiling/heatmap/quantification pipeline be built and validated against
real tissue images without depending on multi-gigabyte .svs files or
OpenSlide setup. Real WSI support (via OpenSlide) is a stretch goal that can
reuse the same tiling_inference.py code path unchanged, since it operates on
a plain numpy RGB array regardless of source.
"""

import argparse
import numpy as np
import h5py
import cv2


def build_mosaic(x_path, grid_size=20, patch_size=96, shuffle=True, seed=0):
    """
    x_path: path to a PCam *_x.h5 file
    grid_size: mosaic will be grid_size x grid_size patches
               (e.g. 20 -> 1920x1920 image, roughly comparable in scale
               concept to a small WSI region)
    Returns: HxWx3 uint8 RGB mosaic image
    """
    rng = np.random.default_rng(seed)

    with h5py.File(x_path, "r") as f:
        n_available = f["x"].shape[0]
        n_needed = grid_size * grid_size
        if shuffle:
            indices = rng.choice(n_available, size=n_needed, replace=False)
            indices.sort()  # h5py fancy indexing requires sorted indices
        else:
            indices = np.arange(n_needed)

        patches = f["x"][indices]

    mosaic = np.zeros((grid_size * patch_size, grid_size * patch_size, 3), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            mosaic[i * patch_size:(i + 1) * patch_size,
                   j * patch_size:(j + 1) * patch_size] = patches[idx]

    return mosaic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", default="test",
                         help="Which PCam split's patches to stitch into the "
                              "mosaic (torchvision naming: train/val/test).")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--out", default="outputs/synthetic_wsi.png")
    args = parser.parse_args()

    # torchvision's PCAM(download=True) stores files under a 'pcam/'
    # subfolder of whatever root you give it - not flat under data_dir
    # directly. See dataset.py's get_pcam_dataset for the same convention.
    x_path = f"{args.data_dir}/pcam/camelyonpatch_level_2_split_{args.split}_x.h5"

    mosaic = build_mosaic(x_path, grid_size=args.grid_size)
    cv2.imwrite(args.out, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    print(f"Built {mosaic.shape[1]}x{mosaic.shape[0]} synthetic mosaic -> {args.out}")


if __name__ == "__main__":
    main()