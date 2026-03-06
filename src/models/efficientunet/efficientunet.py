import segmentation_models_pytorch as smp
import torch.nn as nn

class EfficientUnet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b3", encoder_weights:str | None="imagenet", decoder_attention_type=None, classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
            activation=None
        )
        
        # --- [新增] 在 Decoder 每一層結尾注入 Dropout 0.4 ---
        for i in range(len(self.model.decoder.blocks)):
            self.model.decoder.blocks[i].add_module("dropout", nn.Dropout2d(p=0.4))
            
        self.bottleneck_dropout = nn.Dropout2d(p=0.4)
    
    def forward(self, x):
        # 1. 提取 Encoder 特徵
        features = self.model.encoder(x)
        
        # 2. 在最強大的特徵層 (最後一層) 套用 Dropout
        features[-1] = self.bottleneck_dropout(features[-1])
        
        # 3. 進入 Decoder 並產生輸出
        decoder_output = self.model.decoder(*features)
        logits = self.model.segmentation_head(decoder_output)
        return logits