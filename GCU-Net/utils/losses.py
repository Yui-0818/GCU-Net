import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1e-8):
    pred = torch.sigmoid(logits)
    inter = 2 * (pred * target).sum() + smooth
    union = pred.sum() + target.sum() + smooth
    return 1 - inter / union


class BCEDiceDSLoss(nn.Module):
    """Boundary-weighted BCE + Dice on the main output, with two auxiliary heads."""

    def __init__(self, weights=(1.0, 0.4, 0.2), bce_weight=0.5):
        super().__init__()
        self.weights = weights
        self.bce_weight = bce_weight

    def forward(self, preds, target, weight=None):
        if torch.is_tensor(preds):
            preds = [preds]
        total = 0.0
        for w, logit in zip(self.weights, preds):
            if logit.shape[-2:] != target.shape[-2:]:
                logit = F.interpolate(logit, target.shape[-2:], mode="bilinear",
                                      align_corners=False)
            bce = F.binary_cross_entropy_with_logits(logit, target, weight=weight,
                                                     reduction="mean")
            total = total + w * (self.bce_weight * bce
                                 + (1 - self.bce_weight) * dice_loss(logit, target))
        return total / sum(self.weights[:len(preds)])
