"""U-Net variants backed by `segmentation_models_pytorch` encoders.

Both ResUnet and EfficientUnet are thin wrappers around ``smp.Unet`` so that
encoder choice, attention and dropout are configured in one place.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


class _SmpUnet(nn.Module):
    """Base wrapper around smp.Unet with optional decoder dropout."""

    def __init__(self, encoder_name, encoder_weights=None, attention=None,
                 classes=1, decoder_dropout=0.0):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            decoder_attention_type=attention,
            activation=None,
        )
        self.decoder_dropout = decoder_dropout
        if decoder_dropout > 0:
            for block in self.model.decoder.blocks:
                block.add_module("dropout", nn.Dropout2d(p=decoder_dropout))
            self.bottleneck_dropout = nn.Dropout2d(p=decoder_dropout)
        else:
            self.bottleneck_dropout = nn.Identity()

    def forward(self, x):
        features = self.model.encoder(x)
        features[-1] = self.bottleneck_dropout(features[-1])
        decoded = self.model.decoder(features)
        return self.model.segmentation_head(decoded)


class ResUnet(_SmpUnet):
    """ResNet-encoder U-Net."""

    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet",
                 attention="scse", classes=1, decoder_dropout=0.0):
        super().__init__(encoder_name, encoder_weights, attention, classes,
                         decoder_dropout)


class EfficientUnet(_SmpUnet):
    """EfficientNet-encoder U-Net (default decoder dropout 0.4)."""

    def __init__(self, encoder_name="efficientnet-b3", encoder_weights="imagenet",
                 attention="scse", classes=1, decoder_dropout=0.4):
        super().__init__(encoder_name, encoder_weights, attention, classes,
                         decoder_dropout)
