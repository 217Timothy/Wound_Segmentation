import sys
import torch
from ..core import MODEL_VERSION, DEVICE
from src.models import UNet, ResUnet, EfficientUnet
from src.utils import load_checkpoint


class SegmentationModelManager:
    _models = {}
    
    @classmethod
    def get_model(cls, version: str):
        if version not in MODEL_VERSION:
            raise ValueError(f"Unknown version: {version}")
        
        if version not in cls._models:
            ckpt_path = MODEL_VERSION[version]
            
            if version == "v1":
                model = UNet(n_channels=3, n_classes=1)
                
            elif version == "v2":
                model = ResUnet(
                    encoder_name="resnet50",
                    encoder_weights=None,
                    decoder_attention_type="scse",
                    classes=1
                )
                
            elif version == "v3":
                model = EfficientUnet(
                    encoder_name="efficientnet-b3",
                    encoder_weights=None,
                    decoder_attention_type="scse",
                    classes=1
                )
                
            else:
                raise ValueError(f"[Error] Unsupported Version: {version}")
            
            model.to(DEVICE)
            load_checkpoint(ckpt_path, model)
            model.eval()
            
            cls._models[version] = model
            print(f"✅ [SUCCESS] Model {version} loaded and cached.")
        
        return cls._models[version]