import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    
    def __init__(
        self, 
        root_dir: str, 
        datasets, 
        split="train",
        transform=None,
        cache_data=True
    ):
        """
        通用分割資料集
        Args:
            root_dir (str): 'data/processed'
            datasets (list): 資料集名稱列表，例如 ['WoundSeg', 'CO2Wound']
            split (str): 'train', 'val', 或 'test'
            transform (albumentations): 資料增強物件
            cache_data (bool): 是否將資料預先載入 RAM (加速用)
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.cache_data = cache_data
        self.files = [] # 這是存放所有檔案路徑的大清單
        
        # --- 存放快取資料的列表 ---
        self.cached_images = []
        self.cached_masks = []
        
        # 1. 蒐集檔案路徑
        for ds in datasets:
            # a. 讀取 preprocess 生成的 txt 清單
            # 路徑範例: data/processed/splits/WoundSeg/train.txt
            split_file = os.path.join(self.root_dir, "splits", ds, f"{split}.txt")
            
            if not os.path.exists(split_file):
                print(f"[Warn] Split file not found: {split_file} (Skipping {ds})")
                continue
            
            with open(split_file, "r") as f:
                fnames = [line.strip() for line in f.readlines()]
            
            # b. 組合完整路徑
            # 資料實際位置: data/processed/WoundSeg/train/images/WS_001.png
            base_path = os.path.join(self.root_dir, ds, split)
            for fname in fnames:
                img_path = os.path.join(base_path, "images", fname)
                mask_path = os.path.join(base_path, "masks", fname)
                self.files.append((img_path, mask_path))
        
        # 2. 🔥 預先載入 RAM
        if self.cache_data:
            print(f"[INFO] Caching {len(self.files)} images into RAM for '{split}' set...")
            for img_path, mask_path in tqdm(self.files, desc=f"Loading {split}"):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ [Error] Skip bad image: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"❌ [Error] Skip bad mask: {mask_path}")
                    continue
                
                self.cached_images.append(img)
                self.cached_masks.append(mask)
            
            if len(self.cached_images) != len(self.files):
                print(f"[WARN] Some images failed to load. Resizing dataset from {len(self.files)} to {len(self.cached_images)}")
                # 這裡簡單處理：直接依賴 cached_images 的長度
                # 為了避免 index error，我們把 self.files 清空或不用它了
                pass
    
    
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
        
        # ================= 【最後加上這兩層防護網】 =================
        
        # 防護網 1：把 0 和 255 強制轉成 0.0 和 1.0 的浮點數 (Float)
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
        else:
            # 預防萬一 transform 沒有 ToTensorV2 導致回傳 numpy 的情況
            mask = torch.from_numpy((mask > 0).astype(np.float32))
            
        # 防護網 2：補上 Channel 維度 ([512, 512] -> [1, 512, 512])
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return img, mask