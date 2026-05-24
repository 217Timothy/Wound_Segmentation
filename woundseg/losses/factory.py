"""Loss factory — maps a config name to a loss module."""

from .combo import BCEDiceLoss, BCETverskyLoss, FocalTverskyLoss
from .dice import DiceLoss
from .tversky import TverskyLoss

_BUILDERS = {
    "bce_dice": lambda **kw: BCEDiceLoss(bce_weight=kw.get("bce_weight", 0.5)),
    "bce_tversky": lambda **kw: BCETverskyLoss(
        alpha=kw.get("alpha", 0.4), beta=kw.get("beta", 0.6),
        bce_weight=kw.get("bce_weight", 0.5)),
    "focal_tversky": lambda **kw: FocalTverskyLoss(
        alpha=kw.get("alpha", 0.4), beta=kw.get("beta", 0.6),
        gamma=kw.get("gamma", 1.33)),
    "dice": lambda **kw: DiceLoss(),
    "tversky": lambda **kw: TverskyLoss(
        alpha=kw.get("alpha", 0.4), beta=kw.get("beta", 0.6)),
}


def build_loss(name: str, **kwargs):
    """Return a loss module by name (see keys of `_BUILDERS`)."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown loss {name!r}; choose from {sorted(_BUILDERS)}.")
    return _BUILDERS[key](**kwargs)
