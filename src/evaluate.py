"""Evaluation: precision, recall, AUC, confusion matrix, ROC curve."""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np

from dataset import PCamDataset, get_transforms
from model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of "tumor" class
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--weights", default="outputs/model.pth")
    parser.add_argument("--subset_size", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = PCamDataset(
        f"{args.data_dir}/camelyonpatch_level_2_split_test_x.h5",
        f"{args.data_dir}/camelyonpatch_level_2_split_test_y.h5",
        transform=get_transforms(train=False),
        subset_size=args.subset_size,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    labels, probs = get_predictions(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    print("Saved ROC curve to outputs/roc_curve.png")


if __name__ == "__main__":
    main()
