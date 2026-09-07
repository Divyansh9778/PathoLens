# Architecture & Design Decisions

## Pipeline

```
Large image (synthetic WSI-scale mosaic, see Scope Decision below)
    │
    ▼
Tissue detection (src/tissue_detection.py)
    Otsu threshold on HSV saturation channel + morphological cleanup.
    Standard, well-established technique for separating stained tissue
    from blank slide background in digital pathology.
    │
    ▼
Tile position generation (src/tiling_inference.py: get_tissue_positions)
    Sliding window with 50% overlap, restricted to tiles with sufficient
    tissue content — avoids wasting inference on empty background.
    │
    ▼
Batched CNN inference (src/tiling_inference.py: tile_and_predict)
    ResNet18 (transfer learning from ImageNet), fine-tuned on PatchCamelyon.
    Batched rather than one-tile-at-a-time for throughput (see
    src/benchmark.py for the measured difference).
    │
    ▼
Heatmap reconstruction
    Overlapping tile predictions are averaged per-pixel to produce a
    smooth full-resolution probability heatmap.
    │
    ▼
Quantification (src/quantification.py)
    - Tissue coverage %
    - Estimated tumor area % (of tissue, not of whole image — background
      shouldn't dilute the statistic)
    - Ranked list of most suspicious regions by mean patch probability
    │
    ▼
Interactive viewer (app/server.py + app/static/index.html)
    FastAPI backend serves the image, heatmap overlay, and region data.
    OpenSeadragon (plain JS, no React) handles pan/zoom on the frontend.
    Clicking a region queries the nearest analyzed tile and shows its score.
```

## Scope decision: synthetic mosaic instead of real WSI files

Real whole-slide images (.svs, read via OpenSlide) are frequently 1–4GB
per file and introduce format/IO complexity orthogonal to the CV/ML work
this project demonstrates. Given a 10-day build window, the MVP instead
builds a large image by tiling real PatchCamelyon patches into a mosaic
(`src/mosaic.py`). This is:

- **Honest** — clearly documented, not disguised as a real slide.
- **Sufficient** — every downstream stage (tissue detection, tiling,
  batched inference, heatmap, quantification, viewer) operates on a plain
  RGB numpy array and doesn't care where it came from.
- **Extensible** — real OpenSlide/.svs support can be added later by
  swapping the image-loading step for an OpenSlide reader that yields
  the same RGB array interface; no other code needs to change.

## Why no React

OpenSeadragon is a standalone JavaScript library — it doesn't need a
component framework to provide pan/zoom/deep-zoom functionality. Adding
React would mean a build pipeline, bundler config, and state management
for a UI that's fundamentally: one image viewer, one sidebar of stats,
one click handler. Plain HTML/JS keeps that simple and leaves more of the
10 days for the ML/CV work the JD is actually about.

## Why batched inference is benchmarked, not just used

Anyone can call a model in a loop. Measuring the actual throughput
difference between single-tile and batched inference (and CPU vs GPU, if
available) demonstrates the same systems/performance instincts evident in
the rest of this candidate's background (multithreaded server work,
cache-aware processor design) applied to an ML inference workload — this
is the intended bridge between the existing skill set and the new domain.

## Known limitations (state these honestly in an interview)

- Trained on PatchCamelyon (lymph node metastasis patches), not directly
  on whatever tissue type/stain a specific deployment would see — a real
  system would need retraining/validation per tissue type.
- The synthetic mosaic has no biological continuity across patch
  boundaries (each patch comes from a different real image), so it should
  not be described as a real slide, only as a stand-in for scale/pipeline
  testing.
- No clinical validation. This is explicitly a research/educational
  prototype, not a diagnostic tool.
