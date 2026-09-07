"""ResNet18-based binary classifier for PCam patches."""

import torch.nn as nn
from torchvision import models


def build_model(pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model


if __name__ == "__main__":
    import torch
    model = build_model()
    dummy = torch.randn(4, 3, 96, 96)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [4, 2]
