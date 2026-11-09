import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = r"D:\school_project\IMIP\Experiment2\data\82f3d51b1b37edd4aaba70c66213e60b_5546995784491424214_m(1)\kuiyang"


def _read_ids(img_dir):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir))


def _norm_bounds(img, mean, std):
    x = (img.astype(np.float32) / 255.0 - mean) / std
    return float(x.min()), float(x.max())


def _normalize(img, mean, std, lo, hi):
    x = (img.astype(np.float32) / 255.0 - mean) / std
    x = (x - lo) / (hi - lo + 1e-6)
    return x


def _augment(img, mask):
    if random.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[:, ::-1]
    if random.random() < 0.5:
        img = img[::-1]
        mask = mask[::-1]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
    return img, mask


def _grid(size, patch, stride):
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


class KuiyangDataset(Dataset):
    """Kuiyang corneal ulcer data: img/*.jpg + mask/*.png, where mask==0 is the lesion."""

    def __init__(self, root, split, stats, patch_size=None, stride=None):
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        img_dir = os.path.join(root, split, "img")
        mask_dir = os.path.join(root, split, "mask")
        self.ids = _read_ids(img_dir)
        if not self.ids:
            raise FileNotFoundError(f"no images under {img_dir}")

        self.mean = float(stats["mean"])
        self.std = float(stats["std"])
        self.images = []
        self.masks = []
        self.bounds = []
        for name in self.ids:
            img = cv2.imread(os.path.join(img_dir, name + ".jpg"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                raise FileNotFoundError(f"failed to read pair: {name}")
            self.images.append(img)
            self.masks.append((mask == 0).astype(np.uint8))
            self.bounds.append(_norm_bounds(img, self.mean, self.std))

        self.patches = []
        self.patch_has_lesion = []
        if patch_size is not None:
            for i, img in enumerate(self.images):
                integ = cv2.integral(self.masks[i])
                for y in _grid(img.shape[0], patch_size, stride):
                    for x in _grid(img.shape[1], patch_size, stride):
                        self.patches.append((i, y, x))
                        s = (integ[y + patch_size, x + patch_size] - integ[y, x + patch_size]
                             - integ[y + patch_size, x] + integ[y, x])
                        self.patch_has_lesion.append(bool(s > 0))

    def __len__(self):
        if self.patch_size is not None:
            return len(self.patches)
        return len(self.images)

    def __getitem__(self, idx):
        if self.patch_size is not None:
            i, y, x = self.patches[idx]
            p = self.patch_size
            img = self.images[i][y:y + p, x:x + p]
            mask = self.masks[i][y:y + p, x:x + p]
            img, mask = _augment(img, mask)
        else:
            i = idx
            img = self.images[i]
            mask = self.masks[i]

        lo, hi = self.bounds[i]
        img = _normalize(img, self.mean, self.std, lo, hi)
        img = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return img, mask
