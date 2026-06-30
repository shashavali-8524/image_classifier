# Spot the Fake Photo — Screen Recapture Detector

Detects whether an image is a **real photo** or a **photo of a screen**
(someone re-photographing a phone/laptop/printout instead of the real thing).

## How it works

No deep learning. The detector computes 8 hand-crafted signal-processing
features per image and feeds them into a tiny logistic regression
(8 weights + 1 bias, ~1 KB model file):

- **FFT / frequency features** — re-photographing a screen captures its
  pixel grid, which creates an unusually sharp, regular peak in the image's
  frequency spectrum (moire pattern). Real-world textures don't produce
  this regular periodicity.
- **Sharpness (Laplacian variance)** — recaptures are often subtly
  softer/sharper than the original scene due to the double optical capture.
- **Saturation mean/std** — screens tend to render flatter, less saturated
  colour than light reflecting off real objects.
- **Glare / bright-pixel ratio** — specular highlights off the screen's
  glass surface.
- **Blue colour-cast** — most screens are slightly blue-shifted relative to
  real-world white balance.

These features are model-free to compute; only the final 9-number
logistic regression is "trained," and that takes well under a second.

## Files

| File | Purpose |
|---|---|
| `features.py` | Extracts the 8 features from an image |
| `train.py` | Fits the logistic regression on your `real/` and `screen/` photos, saves `model.json` |
| `predict.py` | Required interface: `python predict.py image.jpg` → prints a 0–1 score |
| `model.json` | Produced by `train.py`; the trained weights `predict.py` loads |
| `requirements.txt` | Python dependencies |
| `NOTE.md` | Approach + accuracy + latency + cost write-up (assignment deliverable) |

## Setup

```bash
pip install -r requirements.txt
```

If your photos are iPhone `.HEIC` files, `pillow-heif` (already in
requirements.txt) lets PIL open them directly — no manual conversion needed.

## 1. Collect data

Take ~50 normal real-world photos and ~50 photos of a screen or printout
(vary lighting, angle, distance, and which screen/printout you use — more
variety makes the model more robust). Put them in two folders:

```
real/
screen/
```

## 2. Train

Edit the `REAL_DIR` / `SCREEN_DIR` paths at the top of `train.py` (or pass
them as arguments), then run:

```bash
python train.py
# or
python train.py "C:/path/to/real" "C:/path/to/screen"
```

This prints a 5-fold cross-validated accuracy estimate (use this honest
number in `NOTE.md`) and saves `model.json` next to the script.

## 3. Predict

```bash
python predict.py some_image.jpg
```

Prints a single number from 0 (real) to 1 (photo of a screen).

`model.json` is always loaded from the same folder as `predict.py`, so this
works regardless of which directory you run the command from — just give it
the image path (relative or absolute), with no extra prefix.

## 4. Measure latency

```bash
python -c "import time; from predict import predict; t=time.time(); predict('real/example.jpg'); print((time.time()-t)*1000, 'ms')"
```

## Troubleshooting

- **`OSError: Invalid argument` on the image path** — make sure you're
  passing only the actual file path, e.g. `python predict.py photo.jpg`,
  not a placeholder string like `path/to/...` left over from an example.
- **Accuracy below 95%** — add more/more varied photos to `real/` and
  `screen/` (different screens, more lighting, more angles) and re-run
  `python train.py`. No code changes are needed.
- **`model.json` not found** — run `train.py` first; `predict.py` needs
  that file to exist in the same folder.