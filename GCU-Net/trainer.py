import csv
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from utils.helpers import dir_exists
from utils.metrics import AverageMeter, binary_metrics, summarize
from utils.losses import dice_loss


def predict_tiled(model, img, tile=512, stride=448):
    assert not model.training, "predict_tiled requires model.eval()"
    _, _, h, w = img.shape
    tile = min(tile, h, w)
    ys = list(range(0, h - tile + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    xs = list(range(0, w - tile + 1, stride))
    if xs[-1] != w - tile:
        xs.append(w - tile)
    prob = torch.zeros((1, 1, h, w), device=img.device, dtype=torch.float32)
    count = torch.zeros((1, 1, h, w), device=img.device, dtype=torch.float32)
    for y in ys:
        for x in xs:
            out = model(img[:, :, y:y + tile, x:x + tile])
            prob[:, :, y:y + tile, x:x + tile] += out.float()
            count[:, :, y:y + tile, x:x + tile] += 1
    return prob / count


class Trainer:
    def __init__(self, model, CFG, loss, train_loader, val_loader):
        self.CFG = CFG
        self.loss = loss
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr,
                                          weight_decay=CFG.weight_decay)
        self.total_epochs = CFG.epochs
        self.warmup = CFG.warmup_epochs
        self.scheduler = LambdaLR(self.optimizer, self._lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(CFG.amp))
        self.checkpoint_dir = os.path.join(CFG.save_dir, CFG.run_name)
        dir_exists(self.checkpoint_dir)
        self.log_path = os.path.join(self.checkpoint_dir, "train_log.csv")
        self.start_epoch = 1
        self.best_dice = 0.0
        self.threshold = CFG.threshold
        cudnn.benchmark = True
        logger.info(f"checkpoint dir: {os.path.abspath(self.checkpoint_dir)}")

    def _lr_lambda(self, epoch):
        if epoch < self.warmup:
            return (epoch + 1) / (self.warmup + 1)
        t = (epoch - self.warmup) / max(1, self.total_epochs - self.warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    def resume(self, path):
        ckpt = torch.load(path, map_location="cuda", weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["lr_scheduler"])
        if self.CFG.amp and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_dice = ckpt.get("best_dice", 0.0)
        logger.info(f"resumed from {path}: epoch {ckpt['epoch']}, best dice {self.best_dice:.4f}")

    def train(self):
        self._init_log()
        for epoch in range(self.start_epoch, self.CFG.epochs + 1):
            tic = time.time()
            train_loss = self._train_epoch(epoch)
            results = {}
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                if results["F1"] > self.best_dice:
                    self.best_dice = results["F1"]
                    self._save_checkpoint(epoch, "best.pth")
                    self._save_val_records(epoch, results)
                    logger.info(f"new best dice {self.best_dice:.4f} at epoch {epoch}")
            self._save_checkpoint(epoch, "last.pth")
            minutes = (time.time() - tic) / 60.0
            self._log_epoch(epoch, train_loss, results, minutes)
            logger.info("EPOCH {:03d} | loss {:.4f} | val_dice {} | {:.1f} min".format(
                epoch, train_loss, f"{results['F1']:.4f}" if results else "-", minutes))

    def _train_epoch(self, epoch):
        self.model.train()
        total = AverageMeter()
        batch_time = AverageMeter()
        tbar = tqdm(self.train_loader, ncols=110, disable=None)
        tic = time.time()
        for img, gt, weight in tbar:
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            weight = weight.cuda(non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(self.CFG.amp)):
                preds = self.model(img)
                loss = self.loss(preds, gt, weight)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total.update(loss.item())
            batch_time.update(time.time() - tic)
            tbar.set_description("TRAIN {} | loss {:.4f} | {:.2f}s/it".format(
                epoch, total.average, batch_time.average))
            tic = time.time()
        self.scheduler.step()
        return float(total.average)

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        self.model.eval()
        records = []
        loss_meter = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=110, disable=None)
        for img, gt, gate, _ in tbar:
            img = img.cuda(non_blocking=True)
            gt_c = gt.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", enabled=bool(self.CFG.amp)):
                logit = predict_tiled(self.model, img, self.CFG.tile_size,
                                      self.CFG.tile_stride)
                loss_meter.update(float((0.5 * F.binary_cross_entropy_with_logits(logit, gt_c)
                                         + 0.5 * dice_loss(logit, gt_c)).item()))
            prob = (torch.sigmoid(logit).float().cpu() * gate).numpy()[0, 0]
            target = gt[0, 0].numpy() > 0.5
            records.append(binary_metrics(prob, target, self.threshold))
        summary = summarize(records)
        logger.info("VAL {:03d} | F1 {:.4f} F1g {:.4f} IOU {:.4f} Sen {:.4f} Spe {:.4f}".format(
            epoch, summary["F1"], summary["F1_global"], summary["IOU"],
            summary["Sen"], summary["Spe"]))
        summary["val_loss"] = float(loss_meter.average)
        summary["records"] = records
        return summary

    def _save_val_records(self, epoch, results):
        rows = []
        ids = [r for r in self.val_loader.dataset.rows]
        for row, rec in zip(ids, results["records"]):
            rows.append([row["id"], f"{rec['F1']:.6f}", f"{rec['IOU']:.6f}",
                         f"{row['lesion_ratio']:.6f}"])
        with open(os.path.join(self.checkpoint_dir, "val_best.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "dice", "iou", "lesion_ratio"])
            w.writerows(rows)
        with open(os.path.join(self.checkpoint_dir, "val_best.txt"), "w", encoding="utf-8") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"threshold: {self.threshold}\n")
            f.write(f"dice: {results['F1']:.6f}\n")
            f.write(f"dice_global: {results['F1_global']:.6f}\n")
            f.write(f"iou: {results['IOU']:.6f}\n")
            f.write(f"iou_global: {results['IOU_global']:.6f}\n")
            f.write(f"auc: {results['AUC']:.6f}\n")
            f.write(f"sensitivity: {results['Sen']:.6f}\n")
            f.write(f"specificity: {results['Spe']:.6f}\n")
            f.write(f"precision: {results['pre']:.6f}\n")
            f.write(f"accuracy: {results['Acc']:.6f}\n")

    def _save_checkpoint(self, epoch, filename):
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.CFG.amp else None,
            "best_dice": self.best_dice,
            "config": {k: (dict(v) if isinstance(v, dict) else v)
                       for k, v in self.CFG.items()},
        }
        torch.save(state, os.path.join(self.checkpoint_dir, filename))

    def _init_log(self):
        if self.start_epoch == 1:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_dice_global",
                     "val_iou", "val_iou_global", "val_auc", "val_sen", "val_spe",
                     "val_pre", "val_acc", "minutes"])
        elif not os.path.exists(self.log_path):
            raise FileNotFoundError(f"resume run is missing {self.log_path}")
        else:
            with open(self.log_path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.reader(f) if r]
            keep = [rows[0]] + [r for r in rows[1:]
                                if int(float(r[0])) < self.start_epoch]
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(keep)

    def _log_epoch(self, epoch, train_loss, results, minutes):
        row = [epoch, f"{self.optimizer.param_groups[0]['lr']:.8g}", f"{train_loss:.6f}"]
        keys = ["val_loss", "F1", "F1_global", "IOU", "IOU_global", "AUC", "Sen",
                "Spe", "pre", "Acc"]
        if results:
            row += [f"{results[k]:.6f}" for k in keys]
        else:
            row += [""] * len(keys)
        row.append(f"{minutes:.2f}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
