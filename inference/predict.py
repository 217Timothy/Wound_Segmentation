import sys
import os 

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np

# 設定路徑 hack 以便引用模組
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.unet.unet import UNet
from datasets.wound_dataset import WoundDataset