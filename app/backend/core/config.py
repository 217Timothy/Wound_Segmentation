import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = ROOT_DIR /"checkpoints"

MODEL_VERSION = {
    "v1": CHECKPOINT_DIR / "v1_best.pt",
    "v2": CHECKPOINT_DIR / "v2_best.pt",
    "v3": CHECKPOINT_DIR / "v3_best.pt",
}

DEFAULT_VERSION = "v2"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

IMG_SIZE = (512, 512)  # 改為 512x512
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)