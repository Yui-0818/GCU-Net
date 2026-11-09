import os
import csv
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance
from utils.metrics import AverageMeter, get_metrics


def predict_tiled(model, img, tile=512, stride=448):
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
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(CFG.amp))
        self.loss = loss
        self.model = model.cuda()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(
            torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        self.checkpoint_dir = os.path.join(CFG.save_dir, CFG["model"]["type"])
        dir_exists(self.checkpoint_dir)
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        self.log_path = os.path.join(self.checkpoint_dir, "train_log.csv")
        self.start_epoch = 1
        self.best_dice = 0.0
        cudnn.benchmark = True
        logger.info(f"checkpoint dir: {os.path.abspath(self.checkpoint_dir)}")

    def resume(self, path):
        ckpt = torch.load(path, map_location="cuda", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if self.CFG.amp and "scaler" in ckpt:
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
                dice = results["F1"]
                if dice > self.best_dice:
                    self.best_dice = dice
                    self._save_checkpoint(epoch, "best.pth")
                    logger.info(f"new best dice {dice:.4f} at epoch {epoch}")
            self._save_checkpoint(epoch, "last.pth")
            if epoch % self.CFG.save_period == 0:
                self._save_checkpoint(epoch, f"checkpoint-epoch{epoch}.pth")
            minutes = (time.time() - tic) / 60.0
            self._log_epoch(epoch, train_loss, results, minutes)
            logger.info(
                "EPOCH {:03d} | train_loss {:.4f} | val_dice {} | {:.1f} min".format(
                    epoch, train_loss,
                    f"{results['F1']:.4f}" if results else "-", minutes))

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=120, disable=None)
        tic = time.time()
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(self.CFG.amp)):
                pre = self.model(img)
                loss = self.loss(pre, gt)
            if self.CFG.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold, with_auc=False).values())
            tbar.set_description(
                "TRAIN ({}) | Loss {:.4f} | F1 {:.4f} IOU {:.4f} Sen {:.4f} Spe {:.4f} | {:.2f}s/it".format(
                    epoch, self.total_loss.average, self.f1.average, self.iou.average,
                    self.sen.average, self.spe.average, self.batch_time.average))
            tic = time.time()
        self.writer.add_scalar("train/loss", self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"train/Learning_rate_{i}", opt_group["lr"], epoch)
        self.lr_scheduler.step()
        return float(self.total_loss.average)

    def _valid_epoch(self, epoch):
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=120, disable=None)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(self.CFG.amp)):
                    if img.shape[-1] > 1024 or img.shape[-2] > 1024:
                        predict = predict_tiled(
                            self.model, img, self.CFG.get("tile_size", 512),
                            self.CFG.get("tile_stride", 448))
                    else:
                        predict = self.model(img)
                    loss = self.loss(predict, gt)
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
        self.writer.add_scalar("val/loss", self.total_loss.average, epoch)
        for k, v in self._metrics_ave().items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        return {"val_loss": float(self.total_loss.average), **self._metrics_ave()}

    def _save_checkpoint(self, epoch, filename):
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.CFG.amp else None,
            "best_dice": self.best_dice,
            "config": self.CFG,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)
        logger.info(f"saved {path}")
        return path

    def _init_log(self):
        if self.start_epoch == 1:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_dice",
                                 "val_iou", "val_auc", "val_sen", "val_spe",
                                 "val_pre", "val_acc", "minutes"])
        elif not os.path.exists(self.log_path):
            raise FileNotFoundError(f"resume run is missing {self.log_path}")

    def _log_epoch(self, epoch, train_loss, results, minutes):
        lr = self.optimizer.param_groups[0]["lr"]
        row = [epoch, f"{lr:.8g}", f"{train_loss:.6f}"]
        if results:
            row += [f"{results['val_loss']:.6f}", f"{results['F1']:.6f}",
                    f"{results['IOU']:.6f}", f"{results['AUC']:.6f}",
                    f"{results['Sen']:.6f}", f"{results['Spe']:.6f}",
                    f"{results['pre']:.6f}", f"{results['Acc']:.6f}"]
        else:
            row += [""] * 8
        row.append(f"{minutes:.2f}")
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):
        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average,
        }
