# Build Log

Notes to self as I go — what I did, what broke, what I learned. Keep this
honest and specific; it's also useful material for interview prep later.

## Day 1

Started with fundamentals (NumPy/PyTorch array vs tensor basics, channels
ordering, GPU tensors), then moved into real project work — ended up
covering roughly 3 planned days' worth of work in one long session, mostly
because infrastructure problems ate a lot of time and had to be solved
before any of the ML work could happen at all.

**Environment / repo (took much longer than expected):**
- [x] Repo scaffolded, environment set up
- [x] Reviewed and refined project plan against JD (PathoLens spec) —
      settled scope: synthetic mosaic instead of real WSI files, plain
      JS viewer instead of React, fallback to Streamlit if the FastAPI
      viewer runs out of time
- [x] Git/GitHub setup went badly wrong (nested `.git` from running
      `git init` one directory too high) and had to be fully torn down
      and rebuilt from a clean local folder — real lesson: verify
      `pwd` and `git remote -v` before assuming git state is what you
      think it is
- [x] Rewrote `dataset.py` to use torchvision's built-in `PCAM` dataset
      instead of manual h5py loading — cleaner, and fixed a split-naming
      mismatch (`val`/`test`, not `train`/`valid`) in the process
- [x] Installed missing NVIDIA driver locally (`nvidia-driver-595-open`)
      after diagnosing via `nvidia-smi`/`lspci`; ran out of local disk
      space (56GB root partition, only 4.2GB free at worst point) trying
      to install CUDA + PyTorch locally — cleared cache/snap/apt cruft,
      but ultimately abandoned local GPU training as impractical given
      disk constraints and went back to cloud notebooks

**Fundamentals:**
- [x] NumPy array shape/indexing basics (H,W,C convention)
- [x] PyTorch tensor vs NumPy array: channels-first (C,H,W) layout,
      autograd/gradient tracking, `torch.cuda.is_available()`

**Training (took 4 attempts total before getting a result that stuck):**
- [x] Attempt 1 (Colab, no regularization): trained ResNet18 transfer
      learning on PCam `val` split (20k patches, 10 epochs). Result:
      93.2% precision, 59.4% recall @ 0.5, AUC 0.894. Diagnosed
      overfitting — train_acc hit 98.9% while val_loss climbed
      0.76→1.61; recall plateaued at ~72% even at aggressive thresholds,
      meaning the model was confidently wrong on real tumor cases, not
      just miscalibrated
- [x] Added `weight_decay` (L2 regularization) to `train.py` based on
      that diagnosis
- [ ] Attempts 2 and 3 (Colab): both actually succeeded, but the trained
      model file got lost before evaluation could run — once to a
      Drive-account mismatch (trained under one Google account, evaluated
      against a different account's empty Drive), once to a
      `credential propagation was unsuccessful` auth error that never
      resolved even after multiple fresh-runtime attempts. Real lesson:
      Colab's Drive mount is a genuine single point of failure; don't
      trust a "Saved to Drive" print statement without verifying the file
      is actually there before ending a session
- [x] Switched to Kaggle Notebooks entirely — no Drive/OAuth dependency,
      much more reliable for this workflow. Required phone verification
      to unlock GPU access, then worked cleanly
- [x] Attempt 4 (Kaggle, with weight decay): best val_acc 78.25% (epoch
      2). Evaluated: 92.0% precision, 62.2% recall @ 0.5, **AUC 0.903**
      (up from 0.894). Threshold sweep: recall reaches **84.1%** at
      threshold 0.1 (up from 72.5% ceiling in attempt 1) — real evidence
      weight decay improved calibration, not just accuracy
- [x] Lost a second Kaggle-trained model to an idle-timeout session
      wipe before learning to use "Save Version → Always save output"
      immediately after training completes, not after the full
      pipeline finishes
- [x] Re-ran training a final time (same result range: best val_acc
      77.5%), this time saving the Kaggle version correctly right after

**Pipeline (all run for the first time against the real trained model):**
- [x] Fixed a bug in `mosaic.py` — still used the old flat h5 file path
      from before the `dataset.py` rewrite; updated to match
      torchvision's `pcam/` subfolder structure
- [x] Built synthetic WSI-scale mosaic (1920×1920, stitched from real
      PCam test patches)
- [x] Ran tissue detection (Otsu threshold) — 54.2% tissue coverage
      on the mosaic
- [x] Ran full tiled batched inference + heatmap reconstruction against
      the real trained model — 1328 tissue-containing tiles processed
- [x] Ran quantification — updated the default classification threshold
      from a naive 0.5 to 0.15, based on the real precision/recall
      tradeoff data from evaluation
- [x] Visually inspected tissue mask vs. heatmap outputs — found and
      documented a real limitation: they don't fully agree with each
      other on the synthetic mosaic, because the checkerboard of
      independent patches creates a lot of partial-tissue tiles that
      the classifier (never trained to reason about tissue completeness)
      still confidently scores. Documented in ARCHITECTURE.md as a
      genuine artifact of the synthetic-mosaic scope decision, not a
      bug in either component
- [x] Ran performance benchmark: batched inference gives 2.52× speedup
      on CPU, **5.86× on GPU** (1330 vs 227 tiles/sec) — batching
      matters more on GPU because there's more idle parallelism to
      exploit
- [x] Updated README Results section with all real, confirmed numbers

**Not done yet — deliberately deferred to a fresh session:**
- [ ] FastAPI + OpenSeadragon interactive viewer — code exists
      (`app/server.py`, `app/static/index.html`) but has not been run
      end-to-end yet
- [ ] Real OpenSlide/.svs support (stretch goal, only if time allows
      after everything else is solid)

**Biggest lesson of the day**: infrastructure reliability (git, cloud
GPU auth, Drive sync) cost far more time than any of the actual ML work.
Worth remembering for future projects — verify state cheaply and often
rather than assuming a previous step's success carried through, and
save/checkpoint immediately after anything expensive completes, not
after the next few steps also finish.

## Day 2

## Day 3

## Day 4

## Day 5

## Day 6

## Day 7

## Day 8

## Day 9

## Day 10