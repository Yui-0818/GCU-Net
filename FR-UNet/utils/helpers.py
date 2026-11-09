import os
import random

import numpy as np
import torch


def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
