"""
Hand-crafted, training-free features that distinguish a REAL photo from a
PHOTO-OF-A-SCREEN (recapture).

Why these features work:
- Screens (LCD/OLED) emit light through a fixed pixel grid. Re-photographing
  a screen creates a faint, regular grid/moire pattern that shows up as an
  unusually strong, "peaky" energy ridge in the image's frequency spectrum.
  Real-world textures (skin, fabric, grass, walls) do not produce that kind
  of regular periodicity.
- Screens often introduce a slight blue colour cast (most panels are
  slightly cool/blue compared to real-world white balance) and have flatter,
  less saturated colour than reflected light off real objects.
- Screens commonly produce small bright specular "glare" patches reflecting
  off the glass surface.
- Recaptured images are very slightly softer/blurrier on top of the moire,
  changing local sharpness statistics (captured via Laplacian variance).

None of this requires training labels to compute -- it's pure signal
processing. We DO use ~100 labelled photos to fit a tiny logistic regression
(8 numbers + 1 bias) on top of these features, which is what pushes accuracy
above the 90%s. That fit is fast (<1 second) and the resulting model is a
single small JSON file, no deep learning framework needed at inference time.
"""

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view


def _load_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _fft_features(gray: np.ndarray) -> dict:
    h, w = gray.shape
    g = gray - gray.mean()
    # Hann window reduces edge artifacts in the FFT
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fft2(g * win)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    total = mag.sum() + 1e-9

    def band_energy(lo, hi):
        mask = (r >= lo * rmax) & (r < hi * rmax)
        return mag[mask].sum()

    mid = band_energy(0.05, 0.25)
    high = band_energy(0.25, 0.6)
    vhigh = band_energy(0.6, 1.01)

    # Moire / pixel-grid "peakiness": in real photos, energy in the
    # mid-frequency ring falls off smoothly. A screen's pixel grid creates a
    # sharp peak well above the local average in that ring.
    ring_mask = (r >= 0.08 * rmax) & (r < 0.55 * rmax)
    ring_vals = mag[ring_mask]
    peak_ratio = ring_vals.max() / (ring_vals.mean() + 1e-9)

    return {
        "fft_high_ratio": (high + vhigh) / total,
        "fft_mid_ratio": mid / total,
        "fft_peak_ratio": peak_ratio,
    }


def _laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(gray, 1, mode="reflect")
    windows = sliding_window_view(padded, (3, 3))
    lap = (windows * k).sum(axis=(-1, -2))
    return float(lap.var())


def _color_features(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    sat = (maxc - minc) / (maxc + 1e-9)
    bright_ratio = float((maxc > 0.92).mean())  # glare / specular highlight %
    blue_dominance = float(b.mean() - r.mean())
    return {
        "sat_mean": float(sat.mean()),
        "sat_std": float(sat.std()),
        "bright_ratio": bright_ratio,
        "blue_dominance": blue_dominance,
    }


FEATURE_ORDER = [
    "fft_high_ratio",
    "fft_mid_ratio",
    "fft_peak_ratio",
    "lap_var",
    "sat_mean",
    "sat_std",
    "bright_ratio",
    "blue_dominance",
]


def extract_features(image_path: str, size: int = 512) -> dict:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    gray = _load_gray(img)

    feats = {}
    feats.update(_fft_features(gray))
    feats["lap_var"] = _laplacian_var(gray) / 1000.0  # rescale to a sane range
    feats.update(_color_features(img))
    return feats


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float64)
