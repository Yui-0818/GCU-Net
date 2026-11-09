import argparse
import os

import numpy as np
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader, WeightedRandomSampler

import models
from dataset_kuiyang import KuiyangDataset, DEFAULT_ROOT
from kuiyang_stats import load_or_compute
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def main(CFG, data_path, batch_size, patch_size, stride, workers, steps, pos_ratio, resume):
    seed_torch()
    stats = load_or_compute(data_path, os.path.join(CFG.save_dir, "kuiyang_stats.json"))
    logger.info(f"kuiyang stats: mean={stats['mean']:.6f} std={stats['std']:.6f}")

    train_dataset = KuiyangDataset(data_path, "train", stats, patch_size, stride)
    val_dataset = KuiyangDataset(data_path, "val", stats)
    flags = np.array(train_dataset.patch_has_lesion)
    n_pos = int(flags.sum())
    n_neg = len(flags) - n_pos
    logger.info(f"patches with lesion: {n_pos}/{len(flags)} ({100*n_pos/len(flags):.2f}%)")
    weights = np.where(flags, pos_ratio / max(n_pos, 1), (1 - pos_ratio) / max(n_neg, 1))
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).double(), num_samples=steps * batch_size, replacement=True)
    train_loader = DataLoader(
        train_dataset, batch_size, sampler=sampler, num_workers=workers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, 1, shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False)

    model = get_instance(models, "model", CFG)
    logger.info(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    loss = get_instance(losses, "loss", CFG)
    trainer = Trainer(model=model, loss=loss, CFG=CFG,
                      train_loader=train_loader, val_loader=val_loader)
    if resume:
        trainer.resume(resume)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default=DEFAULT_ROOT, type=str)
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("-ps", "--patch_size", default=256, type=int)
    parser.add_argument("-s", "--stride", default=128, type=int)
    parser.add_argument("-w", "--workers", default=0, type=int)
    parser.add_argument("--steps", default=400, type=int,
                        help="sampled patches per epoch")
    parser.add_argument("--pos-ratio", default=0.5, type=float,
                        help="fraction of lesion patches per batch")
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--val-every", default=None, type=int)
    parser.add_argument("--resume", default=None, type=str,
                        help="path to last.pth for resuming")
    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.val_every is not None:
        CFG.val_per_epochs = args.val_every
    main(CFG, args.dataset_path, args.batch_size, args.patch_size, args.stride,
         args.workers, args.steps, args.pos_ratio, args.resume)
