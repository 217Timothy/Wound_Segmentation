import os 
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from src.models.unet.unet_block import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    # n_channels: 輸入圖片的通道數 (RGB圖片就是 3)
    # n_classes: 輸出的類別數 (傷口分割就是 1)
    # bilinear: 是否使用雙線性插值來放大圖片 (建議 True)
    
    def __init__(self, n_channels, n_classes, bilinear=True) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        # --- Decoder ---
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        
        # --- Output ---
        self.outc = (OutConv(64, n_classes))
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return logits