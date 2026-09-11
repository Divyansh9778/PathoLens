"""
FastAPI backend for the interactive viewer.

Serves the source image, the heatmap overlay, and per-region quantification
data. Run with: uvicorn app.server:app --reload --port 8000
Then open http://localhost:8000
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import build_model
from tiling_inference import tile_and_predict, overlay_and_save
from quantification import compute_quantification

app = FastAPI(title="PathoLens API")

STATE = {"image": None, "heatmap": None, "tile_probs": None, "stats": None}
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


@app.on_event("startup")
def load_and_run():
    """
    For the MVP, runs analysis once at startup on a fixed image so the
    viewer has something to show. Extend with an upload endpoint if time
    allows.
    """
    weights_path = os.path.join(OUTPUTS_DIR, "model_v3.pth")
    image_path = os.path.join(OUTPUTS_DIR, "synthetic_wsi.png")

    if not (os.path.exists(weights_path) and os.path.exists(image_path)):
        print("Model weights (model_v3.pth) or synthetic_wsi.png not found — "
              "run train.py and mosaic.py first. Viewer will show no data.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    heatmap, tile_probs = tile_and_predict(model, img_rgb, device)
    overlay_and_save(img_rgb, heatmap, os.path.join(OUTPUTS_DIR, "heatmap_overlay.png"))
    stats = compute_quantification(img_rgb, heatmap, tile_probs)

    STATE["image"] = image_path
    STATE["heatmap"] = os.path.join(OUTPUTS_DIR, "heatmap_overlay.png")
    STATE["tile_probs"] = tile_probs
    STATE["stats"] = stats


@app.get("/api/stats")
def get_stats():
    if STATE["stats"] is None:
        return JSONResponse({"error": "No analysis run yet"}, status_code=404)
    return STATE["stats"]


@app.get("/api/image")
def get_image():
    if STATE["image"] is None:
        return JSONResponse({"error": "No image loaded"}, status_code=404)
    return FileResponse(STATE["image"])


@app.get("/api/heatmap")
def get_heatmap():
    if STATE["heatmap"] is None:
        return JSONResponse({"error": "No heatmap generated"}, status_code=404)
    return FileResponse(STATE["heatmap"])


@app.get("/api/region")
def get_region(y: int, x: int, size: int = 96):
    """Return the classifier's probability for the tile nearest (y, x),
    for the 'click a region to inspect it' interaction."""
    if STATE["tile_probs"] is None:
        return JSONResponse({"error": "No analysis run yet"}, status_code=404)

    closest = min(
        STATE["tile_probs"],
        key=lambda t: (t[0] - y) ** 2 + (t[1] - x) ** 2,
    )
    return {"y": closest[0], "x": closest[1], "probability": closest[2]}


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")