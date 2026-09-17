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
  the same RGB array interface; no other code needs to change. **Update:
  this has since been implemented — see "Real WSI support" below.**

## Real WSI support (implemented, not just a stretch goal)

The extensibility point above is no longer theoretical. `src/wsi_loader.py`
wraps OpenSlide and provides `load_wsi_region()`, which reads a real
`.svs`/`.tiff` slide file and returns the exact same `HxWx3` uint8 RGB
numpy array that `mosaic.py`'s synthetic image also produces. Both
`tiling_inference.py` and `quantification.py` accept a `--wsi <path>` flag
as a drop-in alternative to `--image <path>` — no other code changes were
needed anywhere in the pipeline, confirming the original design decision.

**Verified against a real slide**: the standard OpenSlide test file
`CMU-1-Small-Region.svs` (a genuine Aperio-format WSI, single pyramid
level, 2220x2967px) was run through the unmodified tissue detection and
tiling pipeline. Tissue detection correctly found 31.3% coverage and,
critically, the tiling step correctly skipped the background — only
actual tissue-containing tiles received the heatmap overlay, background
was left untouched, visually confirming the tissue mask precisely tracked
the real tissue boundary on genuine slide data, not just the synthetic
mosaic's clean checkerboard.

Requires the OpenSlide system library in addition to the Python package:
```bash
apt-get install -y openslide-tools   # Ubuntu/Debian - works on Colab and Kaggle
pip install openslide-python
```

Usage:
```bash
python src/tiling_inference.py --weights outputs/model_v3.pth --wsi path/to/slide.svs --wsi_level 0 --out outputs/wsi_heatmap.png
python src/quantification.py --weights outputs/model_v3.pth --wsi path/to/slide.svs --threshold 0.3
```

**Re-run with the actual trained model (`model_v3.pth`)**: tissue
detection again found 31.26% coverage (deterministic, matches the earlier
random-weights smoke test exactly). The classifier itself flagged 69.14%
of tissue as tumor, many tiles at 100% confidence. **This number should
not be read as a real finding.** `CMU-1-Small-Region.svs` is a skin
tissue cross-section (visible in the region image above), not lymph node
tissue — a completely different tissue type and staining pattern than
PatchCamelyon, which the model was exclusively trained on. This is a
concrete, real illustration of a limitation already noted elsewhere in
this document ("not directly on whatever tissue type/stain a specific
deployment would see") — the pipeline is genuinely WSI-format-compatible,
but the trained classifier's predictions are only meaningful on the
tissue distribution it was actually trained on. A real deployment would
need separate training/validation per tissue type, not just format
compatibility.

**Known scope limit**: `load_wsi_region()` reads a single region at a
chosen pyramid level, not a full slide-scanning strategy across the whole
pyramid. A real production slide's level 0 can be tens of thousands of
pixels per side - reading that entirely into memory is impractical, and
is exactly the class of problem OpenSlide's own tiled/pyramidal file
format exists to solve. Extending this to systematically scan an entire
large slide (rather than one chosen region) would be the natural next
step, but was out of scope for demonstrating that the pipeline is
genuinely WSI-format-compatible.

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

- **A tuned classification threshold does not automatically transfer
  between model versions.** The 0.15 threshold was tuned specifically for
  model_v2's probability distribution. When model_v3 (trained for
  substantially higher recall, per the Results section) was deployed with
  that same 0.15 threshold unchanged, `tumor_area_percent_of_tissue` read
  83.12% — misleadingly high, because v3 assigns higher probability to
  more borderline tiles by design, so an old threshold calibrated for a
  different model's distribution over-flags. Re-running with a threshold
  re-tuned for v3 (0.3) brought this to 71.51%, a large and expected drop.
  There is no ground-truth label for the synthetic mosaic, so 71.51%
  itself isn't validated as "correct" — only that it moved in the
  expected direction once the threshold mismatch was fixed. **Lesson**:
  any classification/quantification threshold is a property of the
  specific model checkpoint it was tuned against, not a fixed constant —
  it must be re-validated (not just carried over) whenever the model
  changes, even to a strictly better version. `quantification.py` now
  takes `--threshold` as an explicit CLI argument for exactly this
  reason, rather than a silent hardcoded default.

- Trained on PatchCamelyon (lymph node metastasis patches), not directly
  on whatever tissue type/stain a specific deployment would see — a real
  system would need retraining/validation per tissue type.
- The synthetic mosaic has no biological continuity across patch
  boundaries (each patch comes from a different real image), so it should
  not be described as a real slide, only as a stand-in for scale/pipeline
  testing.
- No clinical validation. This is explicitly a research/educational
  prototype, not a diagnostic tool.
- **Observed on the actual synthetic mosaic output**: the tissue-detection
  mask and the tumor-probability heatmap don't fully agree with each other.
  Because the mosaic is a checkerboard of independent real patches, many
  tiles are only partially tissue at their edges. `tissue_detection.py`'s
  tile filter (`min_tissue_fraction=0.1`) is permissive enough to let a lot
  of these partial-tissue tiles through to the classifier, which was never
  trained to reason about "how much of this patch is real tissue" — only
  tumor-vs-normal on whatever content it's given. The result: some
  high-probability (red/orange) heatmap regions land in areas the tissue
  mask considers mostly background. On a real WSI, with large contiguous
  tissue regions instead of a patch checkerboard, this disagreement would
  be far less pronounced — it's a genuine artifact of the synthetic-mosaic
  scope decision, not a bug in either the tissue detector or the
  classifier individually.