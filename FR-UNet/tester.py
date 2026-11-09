import argparse
import csv
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from trainer import Trainer, predict_tiled
from utils.helpers import dir_exists
from utils.metrics import get_metrics


class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, out_dir, show=False):
        self.CFG = CFG
        self.loss = loss
        self.test_loader = test_loader
        self.out_dir = out_dir
        self.show = show
        self.model = model.cuda()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint["state_dict"])
        dir_exists(out_dir)
        if show:
            dir_exists(os.path.join(out_dir, "prob"))
        cudnn.benchmark = True

    def test(self):
        self.model.eval()
        self._reset_metrics()
        rows = []
        tbar = tqdm(self.test_loader, ncols=120, disable=None)
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(self.CFG.amp)):
                    if img.shape[-1] > 1024 or img.shape[-2] > 1024:
                        pre = predict_tiled(
                            self.model, img, self.CFG.get("tile_size", 512),
                            self.CFG.get("tile_stride", 448))
                    else:
                        pre = self.model(img)
                    loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                metrics = get_metrics(pre, gt, threshold=self.CFG.threshold)
                self._metrics_update(*metrics.values())
                rows.append([i, f"{loss.item():.6f}", f"{metrics['F1']:.6f}",
                             f"{metrics['IOU']:.6f}", f"{metrics['AUC']:.6f}",
                             f"{metrics['Sen']:.6f}", f"{metrics['Spe']:.6f}",
                             f"{metrics['pre']:.6f}", f"{metrics['Acc']:.6f}"])
                if self.show:
                    prob = torch.sigmoid(pre)[0, 0].float().cpu().numpy()
                    np.save(os.path.join(self.out_dir, "prob", f"prob_{i:03d}.npy"), prob)
                tbar.set_description(
                    "TEST ({}) | Dice {:.4f} IOU {:.4f}".format(
                        i, self.f1.average, self.iou.average))

        csv_path = os.path.join(self.out_dir, "test_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "loss", "dice", "iou", "auc",
                             "sen", "spe", "pre", "acc"])
            writer.writerows(rows)
        logger.info(f"per-image metrics saved: {csv_path}")
        logger.info(f"test loss: {self.total_loss.average}")
        for k, v in self._metrics_ave().items():
            logger.info(f"{k:5s}: {v}")
