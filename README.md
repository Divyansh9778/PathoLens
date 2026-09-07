# PathoLens — AI-Assisted Histopathology Image Analysis

**Status: research/educational prototype. Does not diagnose disease or
replace a medical professional.**

An end-to-end pipeline that takes a large histopathology image, detects
tissue regions, extracts patches, classifies each patch for tumor presence
with a CNN, reconstructs a slide-level probability heatmap, ranks suspicious
regions, and reports quantitative tissue statistics — visualized through an
interactive zoomable viewer.

## Why this project

Most "medical AI" demo projects stop at a patch-level classifier trained on
a clean, pre-cut dataset. Real digital pathology systems have to deal with
images far too large to feed into a network directly (tens of thousands of
pixels per side), separate actual tissue from blank slide background, turn
per-patch predictions back into something spatially meaningful, and do all
of this fast enough to be useful. This project builds that full pipeline,
not just the classifier at its center.

## Scope decision (read this first)

Real whole-slide image files (.svs, via OpenSlide) are often 1–4GB each and
introduce format-handling complexity that has nothing to do with the ML/CV
skills this project is meant to demonstrate. Given a 10-day timeline, the
MVP pipeline runs against a **synthetic large image**: a mosaic stitched
from real PatchCamelyon patches at true whole-slide-like dimensions. This
lets every stage of the pipeline — tissue detection, tiling, inference,
heatmap reconstruction, quantification — be built and validated honestly.
**Real OpenSlide/.svs support is an explicit stretch goal**, attempted only
if the MVP is solid with time to spare (see fallback plan below).

## Problem

Binary classification: does a 96x96 histopathology patch contain tumor
tissue (metastatic) or healthy tissue?

## Dataset

[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) — 327,680 color
patches (96x96px) extracted from histopathology scans of lymph node
sections, labeled for presence of metastatic tissue. A subset is used here
to keep training tractable.

## Pipeline

```
Large image (synthetic WSI-scale mosaic)
        │
        ▼
Tissue detection (Otsu threshold on HSV saturation — drop blank background)
        │
        ▼
Patch extraction (only over detected tissue, overlapping tiles)
        │
        ▼
CNN inference (ResNet18, transfer learning) — batched, with a benchmarked
        │        single-vs-batched / CPU-vs-GPU comparison
        ▼
Patch classification + Grad-CAM (per-patch interpretability)
        │
        ▼
Heatmap reconstruction (stitch per-tile probabilities back to full image)
        │
        ▼
Suspicious region ranking + quantification (tissue %, tumor-area %,
        │        top-N regions by mean probability)
        ▼
Interactive viewer (FastAPI + OpenSeadragon: pan/zoom, heatmap overlay,
                     click a region to inspect the underlying patch + score)
```

## Approach

1. **Baseline classifier** — ResNet18 (ImageNet-pretrained), fine-tuned on
   PCam patches.
2. **Evaluation** — precision, recall, F1, AUC, confusion matrix (not just
   accuracy — false negatives matter more in a diagnostic setting).
3. **Explainability** — Grad-CAM to visualize which regions of a patch drove
   the prediction.
4. **Tissue detection** — Otsu thresholding on the HSV saturation channel to
   separate tissue from blank slide background before wasting inference on
   empty tiles (a real preprocessing step in digital pathology).
5. **WSI-scale tiling inference** — tile the large image into overlapping
   patches over tissue regions only, classify each tile, stitch results into
   a full-image heatmap.
6. **Quantification** — tissue area %, estimated tumor area %, ranked list
   of most suspicious regions by mean patch probability.
7. **Performance benchmarking** — single-image vs batched inference
   throughput, and CPU vs GPU if available. This is the piece that connects
   your systems background to the ML work — most beginner projects skip it
   entirely.
8. **Interactive viewer** — FastAPI backend serving the image, heatmap
   overlay, and per-region data; a lightweight HTML/JS frontend using
   OpenSeadragon for pan/zoom (no React — it adds build complexity without
   adding capability here). Click a region → see the patch, predicted class,
   and confidence.

## Project structure

```
histo-classifier/
├── src/
│   ├── dataset.py             # PCam loading & preprocessing
│   ├── model.py                # ResNet18 classifier definition
│   ├── train.py                 # training loop
│   ├── evaluate.py             # metrics: precision/recall/F1/AUC/confusion matrix
│   ├── gradcam.py              # Grad-CAM implementation
│   ├── tissue_detection.py     # Otsu-based tissue vs. background segmentation
│   ├── mosaic.py                # builds the synthetic large "WSI-scale" image
│   ├── tiling_inference.py     # tiling + batched inference + heatmap stitching
│   ├── quantification.py       # tissue %, tumor %, suspicious region ranking
│   └── benchmark.py            # single vs batched, CPU vs GPU timing comparison
├── app/
│   ├── server.py                # FastAPI backend: serves image/heatmap/regions
│   ├── static/
│   │   └── index.html           # OpenSeadragon viewer frontend
│   └── demo_app.py             # (fallback) Streamlit demo — see Fallback Plan
├── notebooks/                  # exploration notebooks
├── data/                       # (not committed — see .gitignore)
├── outputs/                    # trained weights, heatmaps, benchmark results
├── ARCHITECTURE.md             # pipeline + design decisions in more depth
└── PROGRESS.md                  # daily build log
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 10-day plan with fallback checkpoints

Each phase has a checkpoint: if you're behind schedule at that point, cut
scope in the stated order rather than rushing every remaining piece.

| Days | Goal | If behind, cut in this order |
|------|------|-------------------------------|
| 1–3 | Python-for-ML, NumPy, OpenCV, PyTorch, CNN/transfer-learning fundamentals | Reduce depth of theory study, not the coding practice — you learn faster by building |
| 4–6 | Train ResNet18 on PCam subset; evaluate (precision/recall/F1/AUC/confusion matrix) | Shrink subset size / epochs before skipping evaluation metrics |
| 7 | Tissue detection + synthetic mosaic + tiling inference + heatmap | Skip tissue detection (run tiling over the whole image) before skipping the heatmap itself |
| 8 | Quantification + region ranking + performance benchmark | Skip the CPU/GPU comparison before skipping tissue-% / tumor-% quantification |
| 9 | Interactive viewer (FastAPI + OpenSeadragon) | **Fall back to `app/demo_app.py` (Streamlit)** — a working simple viewer beats a broken fancy one |
| 10 | README, ARCHITECTURE.md, screenshots/demo, resume bullet, interview prep | Never cut this — an undocumented project is much weaker in an interview than a documented smaller one |

## Results

_(filled in as training progresses — accuracy / precision / recall / AUC /
sample heatmaps / benchmark numbers go here)_

## Status

🚧 In progress — see [PROGRESS.md](PROGRESS.md) for the daily build log.
