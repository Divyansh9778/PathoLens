"""
Tissue vs. background detection.

Whole-slide images have large regions of blank slide background. Real
digital pathology pipelines detect and skip these before running expensive
inference on them. A standard, well-established approach: convert to HSV,
threshold the saturation channel with Otsu's method — stained tissue has
noticeably higher saturation than white/blank background.

Reference approach used broadly in the digital pathology literature (e.g.
CLAM, Camelyon challenge baselines).
"""

import numpy as np
import cv2


def detect_tissue_mask(rgb_image, blur_kernel=7):
    """
    rgb_image: HxWx3 uint8 RGB array
    Returns: HxW boolean mask, True where tissue is detected
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    blurred = cv2.medianBlur(saturation, blur_kernel)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: fill small holes, remove small speckles
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)


def tissue_percentage(mask):
    return float(mask.sum()) / mask.size * 100.0


def patch_has_tissue(mask, y, x, patch_size, min_tissue_fraction=0.1):
    """Return True if the given patch region contains enough tissue to be
    worth running inference on."""
    region = mask[y:y + patch_size, x:x + patch_size]
    if region.size == 0:
        return False
    return (region.sum() / region.size) >= min_tissue_fraction


if __name__ == "__main__":
    import sys
    img_bgr = cv2.imread(sys.argv[1])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = detect_tissue_mask(img_rgb)
    pct = tissue_percentage(mask)
    print(f"Tissue coverage: {pct:.1f}%")

    overlay = img_rgb.copy()
    overlay[~mask] = overlay[~mask] // 3  # dim background for visualization
    cv2.imwrite("outputs/tissue_mask_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Saved outputs/tissue_mask_overlay.png")
