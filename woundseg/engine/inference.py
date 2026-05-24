"""Inference helpers: load a trained model and predict masks."""

import numpy as np
import torch

from ..models import build_model_from_checkpoint


@torch.no_grad()
def infer_one_image(model, img_tensor, device, threshold: float = 0.5) -> np.ndarray:
    """Predict a binary mask for one preprocessed image tensor.

    Args:
        model: a trained model (switched to eval mode internally).
        img_tensor: normalized (C, H, W) tensor.
        device: compute device.
        threshold: probability cut-off for the foreground class.

    Returns:
        An (H, W) uint8 array with values in {0, 1}.
    """
    model.eval()
    x = img_tensor.unsqueeze(0).to(device)
    probs = torch.sigmoid(model(x))
    pred = (probs > threshold).float()
    return pred.squeeze().cpu().numpy().astype(np.uint8)


def load_inference_model(ckpt_path, device, fallback_cfg: dict | None = None):
    """Load a checkpoint and return a ready-to-use (model, checkpoint) pair.

    The architecture is rebuilt from the checkpoint's stored ``model_cfg``.
    For legacy checkpoints without it, pass ``fallback_cfg`` (a dict with
    keys model/encoder_name/attention/classes).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_from_checkpoint(ckpt, device=device, fallback=fallback_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
