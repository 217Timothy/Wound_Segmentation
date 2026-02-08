import segmentation_models_pytorch as smp
import torch.nn as nn


class SMPUnet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)