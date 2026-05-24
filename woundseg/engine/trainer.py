"""Single-epoch training step."""

import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from ..utils.device import autocast_device_type


def train_one_epoch(model, loader, optimizer, scaler, loss_func, device, epoch,
                    grad_clip: float | None = None) -> float:
    """Run one training epoch and return the mean batch loss.

    Automatic mixed precision is enabled only on CUDA devices.
    """
    model.train()
    running_loss = 0.0

    device_type = autocast_device_type(device)
    use_amp = device_type == "cuda"

    loop = tqdm(loader, desc=f"train epoch {epoch}")
    for imgs, masks, _ in loop:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=use_amp):
            logits = model(imgs)
            loss = loss_func(logits, masks)

        scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / max(len(loader), 1)
