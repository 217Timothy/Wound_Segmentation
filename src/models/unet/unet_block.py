import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # --- 第一層卷積 ---
            # kernel_size=3, padding=1: 確保圖片長寬不變 (512x512 進 -> 512x512 出)
            # bias=False: 因為後面接了 BatchNorm，BN 已經有偏差項了，所以 Conv 不需 Bias (省參數)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d: 歸一化，讓訓練更穩定、收斂更快
            nn.BatchNorm2d(mid_channels),
            # ReLU: 激活函數，引入非線性 (inplace=True 代表直接修改記憶體，省空間)
            nn.ReLU(inplace=True),
            
            # --- 第二層卷積 ---
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下坡 (Encoder): MaxPool 縮小尺寸 -> DoubleConv 提取特徵
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # MaxPool2d(2): 圖片長寬除以 2 (例如 512 -> 256)
            # 就像把圖片縮小看，雖然細節糊了，但能看到整體輪廓
            nn.MaxPool2d(2),
            
            # 縮小後，立刻用 DoubleConv 提取特徵，並把通道數變多 (通常翻倍)
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上坡 (Decoder): 上採樣放大 -> 拼接 (Skip Connection) -> DoubleConv
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # 雙線性插值：像修圖軟體拉大圖片一樣，用數學算出來。平滑、沒雜訊。
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, out_channels // 2)
        else:
            # 轉置卷積：讓 AI 學習怎麼放大。容易有棋盤格雜訊，現在比較少用了。
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: 從下面上來的圖 (比較小，特徵很抽象)
        # x2: 從左邊過來的圖 (Skip Connection，比較大，保留了原本的紋理細節)
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 結合了「語意資訊(x1)」和「細節資訊(x2)」
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)