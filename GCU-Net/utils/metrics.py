import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.sum = np.add(self.sum, np.multiply(val, weight))
            self.count = self.count + weight
            self.avg = self.sum / self.count

    @property
    def average(self):
        return float(self.avg) if self.avg is not None else 0.0


def binary_metrics(prob, target, threshold, need_auc=True):
    pred = prob >= threshold
    tp = float((pred & target).sum())
    tn = float((~pred & ~target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    pre = tp / (tp + fp + 1e-8)
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * pre * sen / (pre + sen + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    if need_auc:
        flat = target.ravel()
        if flat.min() == flat.max():
            auc = 0.5
        else:
            p = prob.ravel()
            step = int(np.ceil(flat.size / 2_000_000))
            auc = roc_auc_score(flat[::step], p[::step])
    else:
        auc = 0.5
    return {"AUC": auc, "F1": f1, "Acc": acc, "Sen": sen, "Spe": spe,
            "pre": pre, "IOU": iou, "tp": tp, "fp": fp, "fn": fn}


def summarize(records):
    keys = ("AUC", "F1", "Acc", "Sen", "Spe", "pre", "IOU")
    out = {k: float(np.mean([r[k] for r in records])) for k in keys}
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    pre = tp / (tp + fp + 1e-8)
    sen = tp / (tp + fn + 1e-8)
    out["F1_global"] = 2 * pre * sen / (pre + sen + 1e-8)
    out["IOU_global"] = tp / (tp + fp + fn + 1e-8)
    return out


def tensor_to_numpy(x):
    return torch.sigmoid(x).detach().float().cpu().numpy()
