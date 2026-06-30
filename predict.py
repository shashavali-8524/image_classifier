"""
Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)

Loads model.json (produced by train.py) and scores the image using
hand-crafted frequency / color / sharpness features -- no deep learning
framework needed at inference time, so it's small and fast enough to run
on a phone.
"""

import json
import sys

import numpy as np

from features import extract_features, feature_vector


def load_model(path="model.json"):
    with open(path) as f:
        return json.load(f)


def predict(image_path: str) -> float:
    model = load_model()
    feats = extract_features(image_path)
    x = feature_vector(feats)
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    w = np.array(model["weights"])
    b = model["bias"]

    xs = (x - mean) / std
    z = xs @ w + b
    score = 1 / (1 + np.exp(-z))
    return float(score)


if __name__ == "__main__":
    print(round(predict(sys.argv[1]), 4))
