"""Model factory.

`build_model` is the single place that knows how to turn a set of config
values into a network. `model_config` produces the matching dict that gets
stored inside checkpoints so inference can rebuild the exact architecture
without guessing from run names.
"""

from .smp_unet import EfficientUnet, ResUnet
from .unet import UNet

_VALID = ("unet", "resunet", "efficientunet")


def build_model(model, encoder_name=None, encoder_weights="imagenet",
                attention="scse", classes=1, decoder_dropout=None, device="cpu"):
    """Construct a segmentation model and move it onto `device`.

    Args:
        model: one of "unet", "resunet", "efficientunet".
        encoder_name: smp encoder name (ignored by plain "unet").
        encoder_weights: "imagenet" or None (use None when loading a checkpoint).
        attention: decoder attention type, e.g. "scse" or None.
        classes: number of output channels.
        decoder_dropout: override the model's default decoder dropout.
        device: target device string.
    """
    model = model.lower()
    if model == "unet":
        net = UNet(n_channels=3, n_classes=classes)
    elif model == "resunet":
        kwargs = dict(encoder_name=encoder_name or "resnet50",
                      encoder_weights=encoder_weights, attention=attention,
                      classes=classes)
        if decoder_dropout is not None:
            kwargs["decoder_dropout"] = decoder_dropout
        net = ResUnet(**kwargs)
    elif model == "efficientunet":
        kwargs = dict(encoder_name=encoder_name or "efficientnet-b3",
                      encoder_weights=encoder_weights, attention=attention,
                      classes=classes)
        if decoder_dropout is not None:
            kwargs["decoder_dropout"] = decoder_dropout
        net = EfficientUnet(**kwargs)
    else:
        raise ValueError(f"Unknown model {model!r}; choose from {_VALID}.")
    return net.to(device)


def model_config(cfg) -> dict:
    """Extract the architecture-defining fields from a training config."""
    return {
        "model": cfg.model,
        "encoder_name": cfg.encoder_name,
        "attention": cfg.attention,
        "classes": 1,
    }


def build_model_from_checkpoint(ckpt: dict, device="cpu", fallback: dict | None = None):
    """Rebuild a model from a checkpoint's stored `model_cfg`.

    For legacy checkpoints without `model_cfg`, `fallback` (a dict of the same
    shape) must be supplied.
    """
    model_cfg = ckpt.get("model_cfg") or fallback
    if model_cfg is None:
        raise ValueError(
            "Checkpoint has no 'model_cfg'; pass --model/--encoder_name/--attention "
            "to specify the architecture for this legacy checkpoint.")
    return build_model(encoder_weights=None, device=device, **model_cfg)
