import segmentation_models_pytorch as smp
import torch.nn as nn


class SMPUnet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", decoder_attention_type=None, classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)