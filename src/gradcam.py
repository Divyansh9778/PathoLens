"""Grad-CAM visualization for the trained classifier.

Uses the `grad-cam` pip package (pip install grad-cam) for a robust,
well-tested implementation rather than reimplementing hooks from scratch.
"""

import argparse
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import build_model


def get_gradcam(model, target_layer, input_tensor):
    """
    model: trained model in eval mode
    target_layer: layer to hook (e.g. model.layer4[-1] for ResNet18)
    input_tensor: normalized input tensor, shape [1, 3, H, W]
    Returns: grayscale CAM heatmap, shape [H, W], values in [0, 1]
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)
    return grayscale_cam[0]


def overlay_heatmap(rgb_image_float, grayscale_cam):
    """rgb_image_float: HxWx3 float array in [0,1]. Returns HxWx3 uint8 overlay."""
    return show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/model.pth")
    parser.add_argument("--image", required=True, help="Path to a patch image")
    parser.add_argument("--out", default="outputs/gradcam_example.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (img_float - mean) / std
    input_tensor = torch.tensor(normalized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    grayscale_cam = get_gradcam(model, model.layer4[-1], input_tensor)
    overlay = overlay_heatmap(img_float, grayscale_cam)

    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM overlay to {args.out}")


if __name__ == "__main__":
    main()
