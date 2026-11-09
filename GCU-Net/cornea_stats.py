import json
import os
import sys

import cv2
import numpy as np

from cornea_roi import fit_cornea, imread


def _ids(root, split):
    img_dir = os.path.join(root, split, "img")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir))


def _signature(ids):
    if not ids:
        return "0"
    return f"{len(ids)}:{ids[0]}:{ids[-1]}"


def compute_meta(root, splits=("train", "val", "test"), verbose=True):
    sums = np.zeros(2, np.float64)
    sq = np.zeros(2, np.float64)
    count = 0.0
    meta = {"stats": {}, "splits": {}, "signature": {}}
    for split in splits:
        ids = _ids(root, split)
        rows = []
        for name in ids:
            bgr = imread(os.path.join(root, split, "img", name + ".jpg"), cv2.IMREAD_COLOR)
            raw = imread(os.path.join(root, split, "mask", name + ".png"), cv2.IMREAD_GRAYSCALE)
            if bgr is None or raw is None:
                raise FileNotFoundError(f"failed to read pair {split}/{name}")
            mask = raw == 0
            info = fit_cornea(bgr)
            x0, y0, x1, y1 = info["crop"]
            info["id"] = name
            info["lesion_ratio"] = float(mask.mean())
            info["lesion_area"] = int(mask.sum())
            rows.append(info)
            if split == "train":
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                g = rgb[..., 1]
                eg = np.maximum(g - np.maximum(rgb[..., 0], rgb[..., 2]), 0)
                for ch, arr in enumerate((g, eg)):
                    v = arr[y0:y1:2, x0:x1:2] / 255.0
                    sums[ch] += v.sum(dtype=np.float64)
                    sq[ch] += np.square(v, dtype=np.float64).sum(dtype=np.float64)
                    if ch == 0:
                        count += v.size
        meta["splits"][split] = rows
        meta["signature"][split] = _signature(ids)
        if verbose:
            valid = sum(1 for r in rows if r["valid"])
            print(f"{split}: {len(rows)} images, ellipse valid {valid}, crop height "
                  f"{np.mean([r['crop'][3] - r['crop'][1] for r in rows]):.0f}")
    mean = sums / count
    std = np.sqrt(np.maximum(sq / count - mean ** 2, 1e-8))
    meta["stats"] = {"g_mean": float(mean[0]), "g_std": float(std[0]),
                     "eg_mean": float(mean[1]), "eg_std": float(std[1])}
    return meta


def load_or_compute(root, path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        need = {"train", "val", "test"}
        sig = meta.get("signature", {})
        if (need.issubset(meta.get("splits", {})) and "stats" in meta
                and all(sig.get(s) == _signature(_ids(root, s)) for s in need)):
            return meta
    meta = compute_meta(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    default_root = r"D:\school_project\IMIP\Experiment2\data\82f3d51b1b37edd4aaba70c66213e60b_5546995784491424214_m(1)\kuiyang"
    root = sys.argv[1] if len(sys.argv) > 1 else default_root
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "cornea_meta.json")
    m = load_or_compute(root, out)
    print("stats:", {k: round(v, 6) for k, v in m["stats"].items()})
    print("saved", out)
