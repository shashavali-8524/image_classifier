"""
Train the screen-recapture detector on your own photos.

Usage:
    python train.py
    python train.py "C:/path/to/real" "C:/path/to/screen"   (override default paths below)

Produces:
    model.json  -- tiny logistic-regression weights used by predict.py

Reports a 5-fold cross-validated accuracy (much more reliable than a single
80/20 split on a small dataset) so you have an honest number for your note.
"""

import glob
import json
import os
import sys

import numpy as np

from features import extract_features, feature_vector, FEATURE_ORDER

# ---- EDIT THESE if you don't want to pass paths on the command line ----
REAL_DIR = r"C:\Users\shashavali\Downloads\Sales\real"
SCREEN_DIR = r"C:\Users\shashavali\Downloads\Sales\screen"
# --------------------------------------------------------------------

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif",
            "*.JPG", "*.JPEG", "*.PNG", "*.HEIC", "*.HEIF")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def build_dataset(real_dir, screen_dir):
    real_files = list_images(real_dir)
    screen_files = list_images(screen_dir)
    print(f"Found {len(real_files)} images in real/  and {len(screen_files)} images in screen/")
    if len(real_files) < 10 or len(screen_files) < 10:
        print("Need at least ~10 images in each folder (ideally ~50) before training.")
        sys.exit(1)

    X, y = [], []
    for label, files in ((0, real_files), (1, screen_files)):
        for p in files:
            try:
                X.append(feature_vector(extract_features(p)))
                y.append(label)
            except Exception as e:
                print(f"  skipped (couldn't read) {p}: {e}")

    return np.array(X), np.array(y)


def train_logreg(X, y, epochs=4000, lr=0.1, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def accuracy(X, y, w, b, thresh=0.5):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    preds = (p >= thresh).astype(int)
    return (preds == y).mean()


def cross_validate(X, y, k=5, l2_options=(1e-4, 1e-3, 1e-2, 1e-1)):
    """Try a few regularization strengths, pick the one with the best
    average k-fold held-out accuracy. Returns that accuracy + chosen l2."""
    n = len(y)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    best_l2, best_acc = None, -1
    for l2 in l2_options:
        accs = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            mean = X[train_idx].mean(axis=0)
            std = X[train_idx].std(axis=0) + 1e-9
            Xtr = (X[train_idx] - mean) / std
            Xte = (X[test_idx] - mean) / std
            w, b = train_logreg(Xtr, y[train_idx], l2=l2)
            accs.append(accuracy(Xte, y[test_idx], w, b))
        avg_acc = float(np.mean(accs))
        print(f"  l2={l2:<8} cv_accuracy={avg_acc:.3f}")
        if avg_acc > best_acc:
            best_acc, best_l2 = avg_acc, l2
    return best_acc, best_l2


def main():
    real_dir = sys.argv[1] if len(sys.argv) > 1 else REAL_DIR
    screen_dir = sys.argv[2] if len(sys.argv) > 2 else SCREEN_DIR

    print(f"real folder:   {real_dir}")
    print(f"screen folder: {screen_dir}")
    print("Extracting features from images (this can take a minute)...")
    X, y = build_dataset(real_dir, screen_dir)

    print("\nCross-validating to pick the most robust model...")
    cv_acc, best_l2 = cross_validate(X, y)
    print(f"\nBest 5-fold cross-validated accuracy: {cv_acc:.1%}  (l2={best_l2})")

    if cv_acc < 0.95:
        print("\n*** Below the 95% target. Add more / more varied photos to")
        print("    real/ and screen/ (different screens, lighting, angles)")
        print("    and re-run this script. ***")

    # refit on ALL data with the best l2 for the final shipped model
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    w, b = train_logreg(Xs, y, l2=best_l2)
    final_train_acc = accuracy(Xs, y, w, b)
    print(f"Final model train accuracy (all data, optimistic): {final_train_acc:.3f}")

    model = {
        "feature_order": FEATURE_ORDER,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
    }
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved {model_path}")

    print("\nPut THIS number in your note (be honest, it's the cross-validated one):")
    print(f"  -> {cv_acc:.1%} (5-fold cross-validation accuracy)")


if __name__ == "__main__":
    main()