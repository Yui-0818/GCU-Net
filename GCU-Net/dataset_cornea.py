import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cornea_roi import cornea_roi, erase_white_annotation, imread

cv2.setNumThreads(1)

DEFAULT_ROOT = r"D:\school_project\IMIP\Experiment2\data\82f3d51b1b37edd4aaba70c66213e60b_5546995784491424214_m(1)\kuiyang"


def _augment(img, mask):
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]
    if random.random() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[::-1]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k, axes=(1, 2))
        mask = np.rot90(mask, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def _boundary_weight(mask, amp=2.0, sigma=3.0):
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return np.ones(mask.shape, np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    d = np.where(m > 0, din, dout)
    w = 1.0 + amp * np.exp(-d / sigma)
    return (w / w.mean()).astype(np.float32)


def _channels(patch_bgr, stats):
    g = patch_bgr[..., 1].astype(np.float32)
    r = patch_bgr[..., 2].astype(np.float32)
    b = patch_bgr[..., 0].astype(np.float32)
    eg = np.maximum(g - np.maximum(r, b), 0.0)
    g = (g / 255.0 - stats["g_mean"]) / stats["g_std"]
    eg = (eg / 255.0 - stats["eg_mean"]) / stats["eg_std"]
    return g, eg


class CorneaPatchDataset(Dataset):
    """Positive/negative patch sampling with green-excess and ellipse ROI channels.

    Index layout: [0, n) draws a lesion-centred patch, [n, 2n) draws a background
    patch. Background patches are drawn only from images with sparse lesions and
    are guaranteed to contain no lesion pixel.
    """

    def __init__(self, root, split, meta, patch=256, uniform_pos=0.3, neg_area_max=0.05,
                 erase_line=False):
        self.root = root
        self.split = split
        self.rows = meta["splits"][split]
        self.stats = meta["stats"]
        self.patch = patch
        self.uniform_pos = uniform_pos
        self.erase_line = erase_line
        self.n = len(self.rows)
        for r in self.rows:
            x0, y0, x1, y1 = r["crop"]
            if patch > min(x1 - x0, y1 - y0):
                raise ValueError(f"patch {patch} larger than crop of {r['id']}")
        pool = [i for i, r in enumerate(self.rows) if r["lesion_ratio"] < neg_area_max]
        self.neg_pool = pool or list(range(self.n))

    def __len__(self):
        return 2 * self.n

    def _read(self, i):
        row = self.rows[i]
        bgr = imread(os.path.join(self.root, self.split, "img", row["id"] + ".jpg"),
                     cv2.IMREAD_COLOR)
        raw = imread(os.path.join(self.root, self.split, "mask", row["id"] + ".png"),
                     cv2.IMREAD_GRAYSCALE)
        if bgr is None or raw is None:
            raise FileNotFoundError(f"failed to read {self.split}/{row['id']}")
        return row, bgr, raw == 0

    def _random_crop(self, x0, y0, x1, y1):
        p = self.patch
        return random.randint(x0, x1 - p), random.randint(y0, y1 - p)

    def _positive_crop(self, row, mask):
        p = self.patch
        x0, y0, x1, y1 = row["crop"]
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if len(ys) == 0:
            return self._random_crop(x0, y0, x1, y1)
        if random.random() < self.uniform_pos:
            x, y = self._random_crop(x0, y0, x1, y1)
            for _ in range(20):
                if mask[y:y + p, x:x + p].sum() >= 4:
                    return x, y
                x, y = self._random_crop(x0, y0, x1, y1)
        k = random.randrange(len(ys))
        cy, cx = int(ys[k]) + y0, int(xs[k]) + x0
        j = p // 8
        cx += random.randint(-j, j)
        cy += random.randint(-j, j)
        x = min(max(cx - p // 2, x0), x1 - p)
        y = min(max(cy - p // 2, y0), y1 - p)
        return x, y

    def _negative_crop(self):
        p = self.patch
        last = None
        for _ in range(4):
            row, bgr, mask = self._read(random.choice(self.neg_pool))
            x0, y0, x1, y1 = row["crop"]
            x, y = self._random_crop(x0, y0, x1, y1)
            for _ in range(24):
                if mask[y:y + p, x:x + p].sum() == 0:
                    return row, bgr, mask, x, y
                x, y = self._random_crop(x0, y0, x1, y1)
            last = (row, bgr, mask, x, y)
        return last

    def __getitem__(self, idx):
        if idx < self.n:
            row, bgr, mask = self._read(idx)
            x, y = self._positive_crop(row, mask)
        else:
            row, bgr, mask, x, y = self._negative_crop()

        p = self.patch
        patch_bgr = bgr[y:y + p, x:x + p]
        patch_mask = mask[y:y + p, x:x + p]
        if self.erase_line:
            patch_bgr = erase_white_annotation(patch_bgr)
        g, eg = _channels(patch_bgr, self.stats)
        roi = np.zeros((p, p), np.float32)
        cx, cy, MA, ma, ang = row["ellipse"]
        cv2.ellipse(roi, ((cx - x, cy - y), (MA, ma), ang), 1.0, -1)

        img = np.stack([g, eg, roi], axis=0)
        m = patch_mask.astype(np.float32)
        img, m = _augment(img, m)
        w = _boundary_weight(m)

        return (torch.from_numpy(img).float(),
                torch.from_numpy(m).float().unsqueeze(0),
                torch.from_numpy(w).float().unsqueeze(0))


class CorneaImageDataset(Dataset):
    def __init__(self, root, split, meta, gate_dilate=10, erase_line=False):
        self.root = root
        self.split = split
        self.rows = meta["splits"][split]
        self.stats = meta["stats"]
        self.gate_dilate = gate_dilate
        self.erase_line = erase_line

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        bgr = imread(os.path.join(self.root, self.split, "img", row["id"] + ".jpg"),
                     cv2.IMREAD_COLOR)
        raw = imread(os.path.join(self.root, self.split, "mask", row["id"] + ".png"),
                     cv2.IMREAD_GRAYSCALE)
        if bgr is None or raw is None:
            raise FileNotFoundError(f"failed to read {self.split}/{row['id']}")
        mask = raw == 0
        if self.erase_line:
            bgr = erase_white_annotation(bgr, row["ellipse"])
        g, eg = _channels(bgr, self.stats)
        roi = cornea_roi(mask.shape, row["ellipse"]).astype(np.float32)
        gate = cornea_roi(mask.shape, row["ellipse"], self.gate_dilate).astype(np.float32)
        img = np.stack([g, eg, roi], axis=0)
        return (torch.from_numpy(img).float(),
                torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(gate).unsqueeze(0),
                row["id"])
