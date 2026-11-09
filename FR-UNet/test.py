import argparse
import os

import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader

import models
from dataset_kuiyang import KuiyangDataset, DEFAULT_ROOT
from kuiyang_stats import load_or_compute
from tester import Tester
from utils import losses
from utils.helpers import get_instance


def main(data_path, weight_path, CFG, out_dir, show):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    stats = load_or_compute(data_path, os.path.join(CFG.save_dir, "kuiyang_stats.json"))
    test_dataset = KuiyangDataset(data_path, "test", stats)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, pin_memory=True)
    model = get_instance(models, "model", CFG)
    loss = get_instance(losses, "loss", checkpoint["config"])
    Tester(model, loss, CFG, checkpoint, test_loader, out_dir, show).test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default=DEFAULT_ROOT, type=str)
    parser.add_argument("-wp", "--weight_path",
                        default=os.path.join("saved", "FR_UNet", "best.pth"), type=str)
    parser.add_argument("-o", "--out_dir", default=os.path.join("saved", "FR_UNet"), type=str)
    parser.add_argument("--show", action="store_true", help="save probability maps")
    args = parser.parse_args()
    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    main(args.dataset_path, args.weight_path, CFG, args.out_dir, args.show)
