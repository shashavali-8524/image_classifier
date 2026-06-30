"""
Train the screen-recapture detector on your own photos.

Usage:
    python train.py

Expects two folders next to this script:
    real/      -- ~50 normal photos of real things
    screen/    -- ~50 photos of a screen/printout showing a picture

Produces:
    model.json  -- tiny logistic-regression weights used by predict.py

Also prints a held-out accuracy estimate (80/20 split) so you have an
honest number to put in your note.
"""

import glob
import json
import os
import random
import sys

import numpy as np

from features import extract_features, feature_vector, FEATURE_ORDER

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC", "*.JPG", "*.JPEG", "*.PNG")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def build_dataset():
    real_files = list_images("real")
    screen_files = list_images("screen")
    if len(real_files) < 10 or len(screen_files) < 10:
        print(f"Found {len(real_files)} real/ and {len(screen_files)} screen/ images.")
        print("Put at least ~50 of each in ./real and ./screen before training.")
        sys.exit(1)

    X, y, paths = [], [], []
    for p in real_files:
        try:
            X.append(feature_vector(extract_features(p)))
            y.append(0)
            paths.append(p)
        except Exception as e:
            print(f"skip {p}: {e}")
    for p in screen_files:
        try:
            X.append(feature_vector(extract_features(p)))
            y.append(1)
            paths.append(p)
        except Exception as e:
            print(f"skip {p}: {e}")

    return np.array(X), np.array(y), paths


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


def main():
    random.seed(0)
    np.random.seed(0)

    print("Extracting features from images (this can take a minute)...")
    X, y, paths = build_dataset()

    # standardize features (zero mean, unit variance) -- makes training stable
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std

    # 80/20 split for an honest held-out accuracy number
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    w, b = train_logreg(Xs[train_idx], y[train_idx])
    train_acc = accuracy(Xs[train_idx], y[train_idx], w, b)
    test_acc = accuracy(Xs[test_idx], y[test_idx], w, b) if len(test_idx) else float("nan")
    print(f"Train accuracy: {train_acc:.3f}  ({len(train_idx)} images)")
    print(f"Held-out accuracy: {test_acc:.3f}  ({len(test_idx)} images)")

    # refit on ALL data for the final shipped model
    w, b = train_logreg(Xs, y)
    final_acc = accuracy(Xs, y, w, b)
    print(f"Final (all-data) train accuracy: {final_acc:.3f}")

    model = {
        "feature_order": FEATURE_ORDER,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
    }
    with open("model.json", "w") as f:
        json.dump(model, f, indent=2)
    print("Saved model.json")
    print("\nPut THIS held-out accuracy number in your note (be honest):")
    print(f"  -> {test_acc:.1%} on a held-out 20% split of your own data")


if __name__ == "__main__":
    main()
