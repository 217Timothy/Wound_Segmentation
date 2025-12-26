import os 
import torch

def save_checkpoint(path, model, optimizer, epoch, best_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_metric": best_metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", None)
    return epoch, best_metric