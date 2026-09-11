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

Picked back up to tackle the last deferred piece: the FastAPI +
OpenSeadragon interactive viewer. This session was almost entirely an
infrastructure fight again — Kaggle's idle-timeout session wipes, same
root cause as yesterday's Colab problems — but ended with the backend
fully verified.

**Kaggle workflow lessons (should have known from Day 1, learned properly
this time):**
- [x] Confirmed: every idle timeout wipes `/kaggle/working` completely —
      code, installed packages, everything. This happens regardless of
      whether it's the same notebook or a new one; there's no setting to
      prevent it on the free tier
- [x] Found the actual fix: attach a notebook's own saved output as an
      **Input** to itself (Add Input → search notebook name → click +).
      This is a one-time setup per notebook — once attached, every future
      session (even after a wipe) can pull the last saved model/data back
      instantly via a simple `cp`, no retraining, no re-attaching
- [x] Learned the difference between the live session's file browser
      (mixes Input + Output, confusing) and a saved **Version's dedicated
      page** (clean Output tab, proper download buttons) — the latter is
      the right place to download files from, not the live editor sidebar
- [x] Real confirmed path for this project's attached input:
      `/kaggle/input/notebooks/sharmaji78/patholens/PathoLens/outputs/`
- [x] Learned: copying the whole project folder from the attached Input
      (`cp -r /kaggle/input/.../PathoLens /kaggle/working/PathoLens`) is
      faster and avoids GitHub entirely when recovering from a session
      wipe — only `pip install` still needs to rerun each time, which is
      unavoidable on the free tier without a custom Docker image

**Bug fixes:**
- [x] Fixed `app/server.py` — was hardcoded to look for `model.pth`,
      needed to be `model_v2.pth` to match the actual trained checkpoint
      filename
- [x] Fixed a bad git commit where the author email was a leftover
      placeholder (`your.real@email.com`) instead of the real one — set
      global git config properly and amended the commit so the pushed
      history is correct

**Downloaded model + mosaic from Kaggle to local machine** — learned that
`torch.save()` output is itself a zip-structured file, so Kaggle's
`.zip`-suffixed download IS the `.pth` file, not a wrapper around it;
unzipping it was an unnecessary extra step (just rename `.zip` → `.pth`).
Ultimately decided not to run the viewer locally at all (no local
torch install) and instead verified everything inside Kaggle directly.

**FastAPI backend — fully verified, all four endpoints tested with real
requests against the live server (not just "the code doesn't crash"):**
- [x] `/api/stats` → 200, real quantification JSON (tissue %, tumor %,
      ranked suspicious regions)
- [x] `/api/image` → 200, correct PNG bytes matching the actual mosaic
      file size exactly
- [x] `/api/heatmap` → 200, correct PNG bytes for the heatmap overlay
- [x] `/api/region?y=..&x=..` → 200, returns a real probability for a
      specific clicked coordinate — this is the endpoint the frontend's
      click-to-inspect feature depends on

**Honest status**: backend is genuinely done and tested. The actual
OpenSeadragon browser frontend (`app/static/index.html`) has NOT been
visually tested — the API it depends on is now proven solid, but nobody
has actually loaded the page and clicked around in a real browser yet.
Don't overclaim this as "viewer complete" — it's "backend complete,
frontend built but visually unverified."

## Day 3

Focused entirely on fixing the overfitting diagnosed on Day 1-2. Real
before/after comparison, not just re-running and hoping for a better
number.

**Kaggle workflow — finally solved properly, not just worked around:**
- [x] Realized the fast recovery cell (copy from attached Input) doesn't
      pull code changes from GitHub — had to add `git fetch` +
      `git reset --hard origin/main` as a standing step, since a plain
      `git pull` broke on a diverged-history conflict from an earlier
      force-push
- [x] Fixed a real bug in the recovery cell itself: `cp -r` (or
      `shutil.copytree`) into an already-existing target folder nests
      instead of replacing — switched to always deleting the target
      first. This bug had gotten baked into a saved Version at one
      point (nested `PathoLens/PathoLens/`); fixed by re-saving after
      cleaning it up
- [x] Set up a persistent package cache (`outputs/packages`, installed
      via `pip install --target=`) so `fastapi`/`uvicorn` don't need
      reinstalling every session — cut recovery time from ~1 minute to
      under 30 seconds. Learned the hard way not to `--target` install
      torch/torchvision too, since Kaggle's base image already has them
      and duplicating them wastes ~1GB+ for nothing

**Overfitting fix — three real levers, tested together:**
- [x] Added partial backbone freezing to `model.py`
      (`freeze_until="layer1"/"layer2"/"layer3"`) — found via a
      parameter-count check that even the most aggressive setting only
      cuts trainable params to ~75%, since ResNet18's parameter mass is
      concentrated in `layer4`. Useful but not a silver bullet on its
      own, worth knowing precisely rather than assuming
- [x] Strengthened training augmentation in `dataset.py` — added full
      90° rotations (label-preserving for histopathology, no canonical
      orientation) and heavier colour/stain jitter
- [x] Downloaded PCam's full "train" split (6.8GB, ~65 seconds on
      Kaggle — the download we avoided on Day 1 to save time turned out
      to be cheap after all) and trained on 100,000 patches instead of
      20,000
- [x] Trained v3 with all three changes together: `--train_split train
      --subset_size 100000 --freeze_until layer2`

**Result: the clearest win of the project.** val_loss stayed flat in a
0.28-0.40 band across all 10 epochs (v2's climbed from 0.62 to 1.2+).
Recall @ 0.5 threshold reached **76.1%** (v1 and v2 were both ~62.2% at
this threshold) — a genuine +13.9 point improvement with almost no
precision cost (91.3% vs 92.0%). AUC improved to **0.936** (from 0.903).
Most importantly, v2's recall had plateaued at 84.1% even at the most
aggressive threshold tested — v3's recall was **still climbing** at the
same threshold (93.5%), meaning the model is now generalizing, not just
being confidently wrong less often. **Confirmed dataset size was the
dominant lever** — more so than regularization (Day 2) or freezing/
augmentation alone.

**Updated `server.py` to point at `model_v3.pth`** as the new best model.
README Results section rewritten with full v1→v2→v3 iteration history,
kept rather than deleted, since it's a real demonstration of diagnosing
a problem and fixing it methodically — better interview material than a
single clean-looking number would have been.

**Still pending**: re-running `tiling_inference.py`/`quantification.py`/
the FastAPI viewer against v3 specifically (they were last verified
against v2).

## Day 4

## Day 5

## Day 6

## Day 7

## Day 8

## Day 9

## Day 10