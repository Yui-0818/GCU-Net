import json
import os

import cv2
import numpy as np


def compute_stats(root):
    img_dir = os.path.join(root, "train", "img")
    names = sorted(os.listdir(img_dir))
    if not names:
        raise FileNotFoundError(f"no training images under {img_dir}")
    total = 0
    s = 0.0
    ss = 0.0
    for name in names:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"failed to read {name}")
        x = img.astype(np.float64) / 255.0
        total += x.size
        s += x.sum()
        ss += (x * x).sum()
    mean = s / total
    var = max(ss / total - mean * mean, 0.0)
    return {"mean": float(mean), "std": float(np.sqrt(var)), "pixels": int(total), "images": len(names)}


def load_or_compute(root, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    stats = compute_stats(root)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
