import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    
    def __init__(
        self, 
        root_dir: str, 
        datasets, 
        split="train",
        transform=None
    ):
        """
        通用分割資料集
        Args:
            root_dir (str): 'data/processed'
            datasets (list): 資料集名稱列表，例如 ['WoundSeg', 'CO2Wound']
            split (str): 'train', 'val', 或 'test'
            transform (albumentations): 資料增強物件
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.files = [] # 這是存放所有檔案路徑的大清單
        
        for ds in datasets:
            # 1. 讀取 preprocess 生成的 txt 清單
            # 路徑範例: data/processed/splits/WoundSeg/train.txt
            split_file = os.path.join(self.root_dir, "splits", ds, f"{split}.txt")
            
            if not os.path.exists(split_file):
                print(f"[Warn] Split file not found: {split_file} (Skipping {ds})")
                continue
            
            with open(split_file, "r") as f:
                fnames = [line.strip() for line in f.readlines()]
            
            # 2. 組合完整路徑
            # 資料實際位置: data/processed/WoundSeg/train/images/WS_001.png
            base_path = os.path.join(self.root_dir, ds, split)
            for fname in fnames:
                img_path = os.path.join(base_path, "images", fname)
                mask_path = os.path.join(base_path, "masks", fname)
                self.files.append((img_path, mask_path))
    
    
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx):
        
        # 1. 拿路徑
        img_path, mask_path = self.files[idx]
        
        # 2. 讀圖片 (轉 RGB)
        img = cv2.imread(img_path)
        # 🔥 [除錯關鍵] 檢查是否讀取失敗
        if img is None:
            raise FileNotFoundError(f"❌ 無法讀取圖片，請檢查路徑是否存在: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. 讀 Mask (轉單層灰階)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        
        # 4. 🔥 Data Augmentation (在這裡做!)
        # 我們使用 albumentations，它會同時處理 image 和 mask
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # 5. 轉 Tensor 與標準化 (Normalization)
        # 圖片: 0-255 -> 0.0-1.0 (float32)
        # Mask: 0-255 -> 0.0-1.0 (float32)
        # 如果 transform 裡沒有 ToTensorV2，我們手動轉
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1)) # (H, W, 3) -> (3, H, W)
            img = torch.from_numpy(img)
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.float32) / 255.0
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            mask = np.expand_dims(mask, axis=0) # (H, W) -> (1, H, W) (增加 Channel 維度)
            mask = torch.from_numpy(mask)

        return img, mask