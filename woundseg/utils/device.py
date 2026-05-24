"""Device selection helper."""

import torch


def get_device(prefer: str | None = None) -> str:
    """Return the best available compute device.

    Args:
        prefer: Optional explicit device ("cuda", "mps", "cpu"). If given and
            available it is used as-is; otherwise the best device is auto-picked.
    """
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_device_type(device: str) -> str:
    """Map a device string to a valid torch.autocast device_type."""
    return "cuda" if str(device).startswith("cuda") else "cpu"
