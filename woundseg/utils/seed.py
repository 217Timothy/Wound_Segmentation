"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix every random source so experiments are reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] all random seeds set to {seed}")
