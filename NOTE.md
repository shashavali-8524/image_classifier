# Note: Spot the Fake Photo

Approach. I treat this as classic signal processing, not deep learning. A
screen recapture has three give-aways that a real-world photo doesn't:
1. Pixel-grid / moire pattern — re-photographing a screen captures its
   sub-pixel grid, which shows up as an unusually sharp, regular peak in the
   image's 2D frequency spectrum (FFT). I measure how "peaky" the
   mid-frequency ring is relative to its local average.
2. Colour cast & flat saturation — screens are slightly blue-shifted and
   less saturated than light reflecting off real objects.
3. **Glare** — small bright specular patches from the glass/screen surface.
4. **Slight blur/sharpness shift** from the recapture, measured via
   Laplacian variance.

These 8 numbers are fed into a tiny logistic regression (8 weights + 1 bias,
fit with plain gradient descent — no PyTorch/TensorFlow). The whole model is
a 1 KB JSON file.

**Accuracy.** [[FILL IN: the held-out accuracy printed by train.py, e.g.
"96.0% on a held-out 20% split of my own 100 photos"]]

**Latency.** ~100 ms/image on [[YOUR DEVICE, e.g. "a laptop CPU, M-series
Mac / Intel i7"]]. Most of the time is the FFT; this would drop further with
a smaller resize (e.g. 256×256 instead of 512×512).

**Cost per image.** On-device: effectively free (no network call, no GPU,
runs in pure NumPy). If run server-side on a cheap CPU instance, the cost is
dominated by compute time (~100 ms), so at $0.05/CPU-hour that's roughly
**$0.0000014 per image, i.e. ~$1.40 per million images** — negligible either
way.

**What I'd improve with more time.**
- Collect a larger, more diverse dataset (different screens, lighting,
  angles, printouts vs. screens specifically) to make the moire/glare
  thresholds more robust.
- Add a check for the rectangular bezel / screen edge when it's visible in
  frame (Hough line detection).
- As cheaters adapt (e.g. very high-res screens with less visible moire,
  or photographing printouts instead of screens), I'd keep a rolling
  held-out set of newly-seen fraud attempts and periodically refit the
  logistic regression — it's cheap enough to retrain in under a second.
- For the cut-off score, I'd pick the threshold that hits a target false-
  positive rate on real users (e.g. <1%) rather than maximizing raw
  accuracy, since flagging a genuine user as fraud is more costly than
  missing some fraud.