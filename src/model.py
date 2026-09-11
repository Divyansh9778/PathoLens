"""ResNet18-based binary classifier for PCam patches."""

import torch.nn as nn
from torchvision import models


def build_model(pretrained=True, freeze_until=None):
    """
    pretrained: start from ImageNet weights (transfer learning)
    freeze_until: how much of the backbone to freeze, to reduce overfitting
        on limited data by cutting the number of trainable parameters.
            None      - train everything (original behaviour)
            "layer1"  - freeze conv1/bn1/layer1 only (mild)
            "layer2"  - freeze through layer2 (moderate)
            "layer3"  - freeze through layer3 (aggressive; only layer4 + fc train)

    Rationale: ImageNet-pretrained early layers learn generic edge/texture
    detectors that transfer fine to histopathology. The later layers are
    where task-specific features get learned. With a limited training set,
    fine-tuning all ~11M parameters gives the model more than enough
    capacity to memorize the training data - which is exactly the
    overfitting we measured (train_acc ~99% while val_loss climbed).
    Freezing early layers cuts trainable parameters substantially without
    giving up the useful pretrained features.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_until is not None:
        # ResNet18 layer order: conv1 -> bn1 -> layer1 -> layer2 -> layer3 -> layer4 -> fc
        freeze_order = ["conv1", "bn1", "layer1", "layer2", "layer3"]
        if freeze_until not in ("layer1", "layer2", "layer3"):
            raise ValueError(
                f"freeze_until must be one of None/'layer1'/'layer2'/'layer3', got {freeze_until!r}"
            )
        cutoff = freeze_order.index(freeze_until)
        to_freeze = freeze_order[: cutoff + 1]

        for name in to_freeze:
            module = getattr(model, name)
            for param in module.parameters():
                param.requires_grad = False

    # Replace final FC layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model


def count_parameters(model):
    """Returns (trainable, total) parameter counts - useful for confirming
    freezing actually took effect."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    import torch

    for setting in [None, "layer1", "layer2", "layer3"]:
        model = build_model(pretrained=False, freeze_until=setting)
        trainable, total = count_parameters(model)
        print(f"freeze_until={str(setting):<8} trainable={trainable:>10,} / {total:,} "
              f"({100 * trainable / total:.1f}%)")

    model = build_model(pretrained=False)
    dummy = torch.randn(4, 3, 96, 96)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")  # expect [4, 2]