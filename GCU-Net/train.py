import argparse
import os

import numpy as np
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader, WeightedRandomSampler

import models
from cornea_stats import load_or_compute
from dataset_cornea import CorneaImageDataset, CorneaPatchDataset, DEFAULT_ROOT
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def build_loaders(data_path, meta, batch_size, patch, workers, steps, pos_ratio,
                  erase_line=False):
    train_ds = CorneaPatchDataset(data_path, "train", meta, patch=patch,
                                  erase_line=erase_line)
    val_ds = CorneaImageDataset(data_path, "val", meta, erase_line=erase_line)
    rows = train_ds.rows
    n = len(rows)
    area = np.array([r["lesion_ratio"] for r in rows], np.float64) + 1e-6
    pos = np.sqrt(area)
    pos = pos / pos.sum() * pos_ratio
    neg = np.full(n, (1 - pos_ratio) / n)
    weights = np.concatenate([pos, neg])
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                    num_samples=steps * batch_size, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=workers, pin_memory=True, drop_last=True,
                              persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=min(2, workers),
                            pin_memory=True)
    logger.info(f"train images {n}, virtual indices {len(train_ds)}, "
                f"positive share {pos.sum():.3f}, steps/epoch {steps}")
    return train_loader, val_loader


CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def main(CFG, args):
    seed_torch(42)
    meta_path = os.path.join(CFG.save_dir, "cornea_meta.json")
    meta = load_or_compute(args.dataset_path, meta_path)
    s = meta["stats"]
    logger.info("channels: g mean {:.4f} std {:.4f} | eg mean {:.4f} std {:.4f}".format(
        s["g_mean"], s["g_std"], s["eg_mean"], s["eg_std"]))
    for split in ("train", "val"):
        rows = meta["splits"][split]
        bad = sum(1 for r in rows if not r["valid"])
        logger.info(f"{split}: {len(rows)} images, invalid ellipse {bad}, "
                    f"mean lesion {np.mean([r['lesion_ratio'] for r in rows]) * 100:.2f}%")

    train_loader, val_loader = build_loaders(
        args.dataset_path, meta, args.batch_size, args.patch_size, args.workers,
        args.steps, args.pos_ratio, bool(CFG.get("erase_line", False)))

    model = get_instance(models, "model", CFG)
    logger.info(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    loss = get_instance(losses, "loss", CFG)
    trainer = Trainer(model=model, CFG=CFG, loss=loss,
                      train_loader=train_loader, val_loader=val_loader)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default=DEFAULT_ROOT, type=str)
    parser.add_argument("-bs", "--batch_size", default=8, type=int)
    parser.add_argument("-ps", "--patch_size", default=256, type=int)
    parser.add_argument("-w", "--workers", default=4, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--pos-ratio", default=0.55, type=float)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--val-every", default=None, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    args = parser.parse_args()

    with open(os.path.join(CODE_DIR, "config.yaml"), encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    CFG.save_dir = os.path.join(CODE_DIR, CFG.save_dir)
    CFG.run_name = args.run_name or CFG.get("run_name", CFG.model["type"])
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.val_every is not None:
        CFG.val_per_epochs = args.val_every
    main(CFG, args)
