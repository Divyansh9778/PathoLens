"""
Performance benchmarking: single-tile vs batched inference, and CPU vs GPU
if available. This is the piece that ties your existing systems/performance
background into the ML project — most beginner CV projects never measure
this at all.
"""

import argparse
import time
import json
import cv2
import torch

from model import build_model
from tiling_inference import get_tissue_positions, get_inference_transform, PATCH_SIZE


def benchmark_single(model, large_image, positions, device, transform, limit=200):
    positions = positions[:limit]
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for y, x in positions:
            tile = large_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            tensor = transform(tile).unsqueeze(0).to(device)
            _ = model(tensor)
    elapsed = time.perf_counter() - start
    return elapsed, len(positions)


def benchmark_batched(model, large_image, positions, device, transform,
                       batch_size=32, limit=200):
    positions = positions[:limit]
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i + batch_size]
            tiles = [large_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] for y, x in batch_positions]
            tensors = torch.stack([transform(t) for t in tiles]).to(device)
            _ = model(tensors)
    elapsed = time.perf_counter() - start
    return elapsed, len(positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--limit", type=int, default=200,
                         help="Number of tiles to use for the benchmark")
    parser.add_argument("--out", default="outputs/benchmark.json")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = get_inference_transform()

    results = {}

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device_name in devices:
        device = torch.device(device_name)
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))

        positions = get_tissue_positions(img_rgb)

        single_time, n1 = benchmark_single(model, img_rgb, positions, device, transform, args.limit)
        batched_time, n2 = benchmark_batched(model, img_rgb, positions, device, transform, 32, args.limit)

        results[device_name] = {
            "single_inference": {
                "tiles": n1,
                "total_seconds": round(single_time, 3),
                "tiles_per_second": round(n1 / single_time, 2),
            },
            "batched_inference": {
                "tiles": n2,
                "total_seconds": round(batched_time, 3),
                "tiles_per_second": round(n2 / batched_time, 2),
            },
            "speedup_from_batching": round(single_time / batched_time, 2),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
