import argparse
import csv
import os

import numpy as np
import torch
from bunch import Bunch
from loguru import logger
from ruamel.yaml import safe_load
from sklearn.metrics import roc_auc_score

import models
from cornea_stats import load_or_compute
from dataset_cornea import CorneaImageDataset, DEFAULT_ROOT
from trainer import predict_tiled
from utils.helpers import get_instance

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(weight):
    ckpt = torch.load(weight, map_location="cuda", weights_only=True)
    cfg = Bunch(ckpt["config"])
    cfg.save_dir = os.path.join(CODE_DIR, "saved")
    cfg.run_name = cfg.get("run_name", cfg.model["type"])
    model = get_instance(models, "model", cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cuda()
    return model, cfg, ckpt


def counts_from(prob, target, thresholds):
    rows = []
    for t in thresholds:
        pred = prob >= t
        rows.append([int((pred & target).sum()), int((pred & ~target).sum()),
                     int((~pred & target).sum())])
    return np.array(rows, np.int64)


@torch.no_grad()
def collect(model, loader, cfg, thresholds):
    records = []
    for img, gt, gate, name in loader:
        img = img.cuda(non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(cfg.amp)):
            logit = predict_tiled(model, img, cfg.tile_size, cfg.tile_stride)
        prob_raw = torch.sigmoid(logit).float().cpu().numpy()[0, 0]
        prob = prob_raw * gate[0, 0].numpy()
        target = gt[0, 0].numpy() > 0.5
        flat = target.ravel()
        auc = 0.5 if flat.min() == flat.max() else roc_auc_score(flat, prob.ravel())
        records.append({
            "id": name[0],
            "lesion_ratio": float(target.mean()),
            "auc": float(auc),
            "counts": counts_from(prob, target, thresholds),
            "counts_raw": counts_from(prob_raw, target, thresholds),
        })
        print(f"  {name[0]} done", flush=True)
    return records


def global_counts(records, index, key="counts"):
    return np.sum([r[key][index] for r in records], axis=0)


def dice_iou(tp, fp, fn):
    return float(2 * tp / (2 * tp + fp + fn + 1e-8)), float(tp / (tp + fp + fn + 1e-8))


def per_image_rows(records, index, key="counts"):
    rows = []
    for r in records:
        tp, fp, fn = r[key][index]
        dice, iou = dice_iou(tp, fp, fn)
        pre = tp / (tp + fp + 1e-8)
        sen = tp / (tp + fn + 1e-8)
        rows.append({"id": r["id"], "dice": dice, "iou": iou, "auc": r["auc"],
                     "sen": sen, "pre": pre, "lesion_ratio": r["lesion_ratio"],
                     "tp": int(tp), "fp": int(fp), "fn": int(fn)})
    return rows


def main(args):
    with open(os.path.join(CODE_DIR, "config.yaml"), encoding="utf-8") as f:
        base_cfg = Bunch(safe_load(f))
    meta = load_or_compute(args.dataset_path, os.path.join(CODE_DIR, base_cfg.save_dir,
                                                           "cornea_meta.json"))
    model, cfg, ckpt = load_model(args.weight)
    out_dir = os.path.join(cfg.save_dir, cfg.run_name)
    logger.info(f"weight epoch {ckpt['epoch']}, config threshold {cfg.threshold}")

    erase = bool(cfg.get("erase_line", False))
    val_loader = torch.utils.data.DataLoader(
        CorneaImageDataset(args.dataset_path, "val", meta, erase_line=erase), 1,
        shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CorneaImageDataset(args.dataset_path, "test", meta, erase_line=erase), 1,
        shuffle=False, num_workers=2)

    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    print("collecting validation probabilities ...", flush=True)
    val_records = collect(model, val_loader, cfg, thresholds)
    val_dice = [dice_iou(*global_counts(val_records, i))[0] for i in range(len(thresholds))]
    val_macro = [float(np.mean([dice_iou(*r["counts"][i])[0] for r in val_records]))
                 for i in range(len(thresholds))]
    best_i = int(np.argmax(val_macro))
    if best_i in (0, len(thresholds) - 1):
        logger.warning("best validation threshold lies on the search boundary")
    chosen = float(thresholds[best_i])
    logger.info(f"threshold selected on val macro dice: {chosen:.2f} -> "
                f"{val_macro[best_i]:.4f} (global {val_dice[best_i]:.4f})")

    with open(os.path.join(out_dir, "val_threshold_sweep.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "dice_global", "iou_global", "dice_global_ungated",
                    "dice_macro"])
        for i, t in enumerate(thresholds):
            d_raw = dice_iou(*global_counts(val_records, i, "counts_raw"))[0]
            d, iou = dice_iou(*global_counts(val_records, i))
            w.writerow([f"{t:.2f}", f"{d:.6f}", f"{iou:.6f}", f"{d_raw:.6f}",
                        f"{val_macro[i]:.6f}"])

    i_chosen = int(np.where(np.isclose(thresholds, chosen))[0][0])
    i_half = int(np.where(np.isclose(thresholds, 0.50))[0][0])
    val_rows = per_image_rows(val_records, i_chosen)
    val_raw = per_image_rows(val_records, i_chosen, "counts_raw")
    with open(os.path.join(out_dir, "val_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dice", "dice_raw", "iou", "auc", "sen", "pre",
                    "lesion_ratio", "tp", "fp", "fn"])
        for r, rr in zip(val_rows, val_raw):
            w.writerow([r["id"], f"{r['dice']:.6f}", f"{rr['dice']:.6f}", f"{r['iou']:.6f}",
                        f"{r['auc']:.6f}", f"{r['sen']:.6f}", f"{r['pre']:.6f}",
                        f"{r['lesion_ratio']:.6f}", r["tp"], r["fp"], r["fn"]])

    print("collecting test probabilities ...", flush=True)
    test_records = collect(model, test_loader, cfg, thresholds)
    test_rows = per_image_rows(test_records, i_chosen)
    test_rows_half = per_image_rows(test_records, i_half)
    test_rows_raw = per_image_rows(test_records, i_chosen, "counts_raw")
    with open(os.path.join(out_dir, "test_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dice", "dice_raw", "iou", "auc", "sen", "pre",
                    "lesion_ratio", "tp", "fp", "fn"])
        for r, rr in zip(test_rows, test_rows_raw):
            w.writerow([r["id"], f"{r['dice']:.6f}", f"{rr['dice']:.6f}", f"{r['iou']:.6f}",
                        f"{r['auc']:.6f}", f"{r['sen']:.6f}", f"{r['pre']:.6f}",
                        f"{r['lesion_ratio']:.6f}", r["tp"], r["fp"], r["fn"]])

    with open(os.path.join(out_dir, "test_threshold_sweep.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# diagnostic only; threshold was selected on validation"])
        w.writerow(["threshold", "dice_global", "iou_global"])
        for i, t in enumerate(thresholds):
            d, iou = dice_iou(*global_counts(test_records, i))
            w.writerow([f"{t:.2f}", f"{d:.6f}", f"{iou:.6f}"])

    tp, fp, fn = global_counts(test_records, i_chosen)
    g_dice, g_iou = dice_iou(tp, fp, fn)
    tp_r, fp_r, fn_r = global_counts(test_records, i_chosen, "counts_raw")
    g_dice_raw, g_iou_raw = dice_iou(tp_r, fp_r, fn_r)
    tph, fph, fnh = global_counts(test_records, i_half)
    g_dice_half, _ = dice_iou(tph, fph, fnh)
    dices = np.array([r["dice"] for r in test_rows])
    dices_raw = np.array([r["dice"] for r in test_rows_raw])
    areas = np.array([r["lesion_ratio"] for r in test_rows]) * 100
    corr = float(np.corrcoef(areas, dices)[0, 1])
    vt, vf, vn = global_counts(val_records, i_chosen)
    val_g_dice, val_g_iou = dice_iou(vt, vf, vn)
    vtr, vfr, vfn = global_counts(val_records, i_chosen, "counts_raw")
    val_g_dice_raw, _ = dice_iou(vtr, vfr, vfn)

    with open(os.path.join(out_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
        f.write("metric region: predictions multiplied by the fitted cornea ellipse "
                "(dilated 10 px); ungated columns are full-image references\n")
        f.write(f"weight: {args.weight}\n")
        f.write(f"epoch: {ckpt['epoch']}\n")
        f.write(f"threshold selected on val macro dice: {chosen:.2f}\n")
        f.write(f"val macro dice at selected threshold: {val_macro[best_i]:.6f}\n")
        f.write(f"val global dice at selected threshold: {val_g_dice:.6f}\n")
        f.write(f"val global iou at selected threshold: {val_g_iou:.6f}\n")
        f.write(f"val global dice ungated: {val_g_dice_raw:.6f}\n")
        f.write(f"test dice mean: {dices.mean():.6f}\n")
        f.write(f"test dice mean ungated: {dices_raw.mean():.6f}\n")
        f.write(f"test dice std: {dices.std():.6f}\n")
        f.write(f"test dice median: {np.median(dices):.6f}\n")
        f.write(f"test iou mean: {np.mean([r['iou'] for r in test_rows]):.6f}\n")
        f.write(f"test auc mean: {np.mean([r['auc'] for r in test_rows]):.6f}\n")
        f.write(f"test sen mean: {np.mean([r['sen'] for r in test_rows]):.6f}\n")
        f.write(f"test pre mean: {np.mean([r['pre'] for r in test_rows]):.6f}\n")
        f.write(f"test global dice: {g_dice:.6f}\n")
        f.write(f"test global iou: {g_iou:.6f}\n")
        f.write(f"test global dice ungated: {g_dice_raw:.6f}\n")
        f.write(f"test global iou ungated: {g_iou_raw:.6f}\n")
        f.write(f"test at 0.50: dice mean {np.mean([r['dice'] for r in test_rows_half]):.6f}, "
                f"global dice {g_dice_half:.6f}\n")
        f.write(f"pearson r(area, dice): {corr:.6f}\n")
        f.write(f"images dice<0.5: {int((dices < 0.5).sum())}\n")
        f.write(f"images dice<0.3: {int((dices < 0.3).sum())}\n")

    logger.info(f"TEST | dice {dices.mean():.4f} (ungated {dices_raw.mean():.4f}) "
                f"iou {np.mean([r['iou'] for r in test_rows]):.4f} | "
                f"global {g_dice:.4f}/{g_iou:.4f} (ungated {g_dice_raw:.4f}/{g_iou_raw:.4f})")
    print(f"selected threshold {chosen:.2f} | test macro dice {dices.mean():.4f} "
          f"(ungated {dices_raw.mean():.4f}) | global dice {g_dice:.4f} "
          f"iou {g_iou:.4f} (ungated {g_dice_raw:.4f}/{g_iou_raw:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--weight", default=os.path.join(CODE_DIR, "saved",
                                                                "GCU_Net_NoLine", "best.pth"))
    parser.add_argument("-dp", "--dataset_path", default=DEFAULT_ROOT)
    args = parser.parse_args()
    main(args)
