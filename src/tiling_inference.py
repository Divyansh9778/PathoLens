"""
WSI-scale tiling inference.

Takes a large image (simulating a whole-slide image), tiles it into
overlapping patches, runs the classifier on each tile, and stitches the
per-tile predictions back into a full-resolution heatmap overlay.

This is the piece that separates "I trained a classifier" from "I built
something that handles the actual scale digital pathology systems deal with."
"""

import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from model import build_model
from tissue_detection import detect_tissue_mask, patch_has_tissue

PATCH_SIZE = 96
STRIDE = 48  # 50% overlap for smoother heatmaps


def get_inference_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def get_tissue_positions(large_image, patch_size=PATCH_SIZE, stride=STRIDE,
                          use_tissue_mask=True):
    """Return list of (y, x) tile top-left positions, optionally skipping
    tiles with insufficient tissue content."""
    h, w, _ = large_image.shape
    all_positions = [
        (y, x)
        for y in range(0, h - patch_size + 1, stride)
        for x in range(0, w - patch_size + 1, stride)
    ]

    if not use_tissue_mask:
        return all_positions

    mask = detect_tissue_mask(large_image)
    return [
        (y, x) for y, x in all_positions
        if patch_has_tissue(mask, y, x, patch_size)
    ]


def tile_and_predict(model, large_image, device, patch_size=PATCH_SIZE,
                      stride=STRIDE, batch_size=32, use_tissue_mask=True):
    """
    large_image: HxWx3 uint8 RGB array
    Returns: (heatmap, positions) — heatmap is HxW per-pixel tumor
             probability built from averaging overlapping tile predictions;
             positions is the list of tile locations actually processed
             (useful for region ranking / quantification downstream).
    """
    h, w, _ = large_image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    transform = get_inference_transform()
    model.eval()

    positions = get_tissue_positions(large_image, patch_size, stride, use_tissue_mask)
    tile_probs = []  # parallel list of (y, x, prob) for quantification/ranking

    with torch.no_grad():
        for i in tqdm(range(0, len(positions), batch_size), desc="tiling inference"):
            batch_positions = positions[i:i + batch_size]
            tiles = [large_image[y:y + patch_size, x:x + patch_size] for y, x in batch_positions]
            tensors = torch.stack([transform(t) for t in tiles]).to(device)

            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for (y, x), prob in zip(batch_positions, probs):
                heatmap[y:y + patch_size, x:x + patch_size] += prob
                counts[y:y + patch_size, x:x + patch_size] += 1
                tile_probs.append((y, x, float(prob)))

    counts[counts == 0] = 1
    heatmap = heatmap / counts
    return heatmap, tile_probs


def overlay_and_save(large_image, heatmap, out_path):
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(large_image, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved heatmap overlay to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/model.pth")
    parser.add_argument("--image", required=True,
                         help="Path to a large image (simulated WSI region)")
    parser.add_argument("--out", default="outputs/wsi_heatmap.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    heatmap, tile_probs = tile_and_predict(model, img_rgb, device)
    overlay_and_save(img_rgb, heatmap, args.out)
    print(f"Processed {len(tile_probs)} tissue-containing tiles")


if __name__ == "__main__":
    main()
