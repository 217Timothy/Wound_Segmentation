import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import Counter


class SegmentationDataset(Dataset):
    
    def __init__(
        self, 
        root_dir: str, 
        datasets, 
        split="train",
        transform=None,
        cache_data=True
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_data = cache_data

        self.files = []  # (img_path, mask_path, dataset_name)
        self.cached_images = []
        self.cached_masks = []

        # ==========================================
        # 1. 收集資料（用 split.txt 控制）
        # ==========================================
        for ds in datasets:
            split_file = os.path.join(root_dir, "splits", ds, f"{split}.txt")

            if not os.path.exists(split_file):
                print(f"[WARN] Split file not found: {split_file}")
                continue

            with open(split_file, "r") as f:
                fnames = [line.strip() for line in f.readlines()]

            base_path = os.path.join(root_dir, ds, split)

            for fname in fnames:
                img_path = os.path.join(base_path, "images", fname)
                mask_path = os.path.join(base_path, "masks", fname)

                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue

                self.files.append((img_path, mask_path, ds))

        print(f"📦 Loaded {len(self.files)} samples ({split})")

        # ==========================================
        # 🔥 印 dataset 分布（超重要）
        # ==========================================
        ds_names = [f[2] for f in self.files]
        print("📊 Dataset distribution:", Counter(ds_names))

        # ==========================================
        # 2. Cache 到 RAM
        # ==========================================
        if self.cache_data:
            print(f"[INFO] Caching {len(self.files)} images into RAM...")

            valid_files = []

            for img_path, mask_path, ds in tqdm(self.files, desc=f"Caching {split}"):

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ Skip bad image: {img_path}")
                    continue

                if mask is None:
                    print(f"❌ Skip bad mask: {mask_path}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.cached_images.append(img)
                self.cached_masks.append(mask)
                valid_files.append((img_path, mask_path, ds))

            self.files = valid_files
            print(f"[INFO] Final cached samples: {len(self.files)}")

    # ==========================================
    def __len__(self):
        return len(self.files)

    # ==========================================
    def __getitem__(self, idx):

        # 🔥 讀資料（cache or disk）
        if self.cache_data:
            img = self.cached_images[idx]
            mask = self.cached_masks[idx]
            ds_name = self.files[idx][2]
        else:
            img_path, mask_path, ds_name = self.files[idx]

            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"❌ Cannot read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"❌ Cannot read mask: {mask_path}")

        # ==========================================
        # 🔥 Augmentation
        # ==========================================
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # ==========================================
        # 🔥 防護網（mask 統一）
        # ==========================================
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
        else:
            mask = torch.from_numpy((mask > 0).astype(np.float32))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # ==========================================
        # 🔥 回傳 dataset label（關鍵）
        # ==========================================
        return img, mask, ds_name


class TKRDataset(Dataset):
    
    def __init__(
        self,
        root_dir,
        ds="TKR",
        split="train",
        transform=None,
        cache_data=True
    ):
        """
        Expected structure:
        root_dir/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        ├── test/
            ├── images/
            └── masks/
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        
        self.split_file = os.path.join(root_dir, ds, "splits", f"{split}.txt")
        self.image_dir = os.path.join(root_dir, ds, split, "images")
        self.mask_dir = os.path.join(root_dir, ds, split, "masks")
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, "r") as f:
            filenames = [line.strip() for line in f.readlines() if line.strip()]
        
        self.files = []
        for fname in filenames:
            stem, _ = os.path.splitext(fname)
            image_path  = os.path.join(self.image_dir, f"{stem}.png")
            mask_path = os.path.join(self.mask_dir, f"{stem}.png")
            self.files.append((image_path, mask_path))
        
        self.cached_images = []
        self.cached_masks = []
        
        if self.cache_data:
            valid_files = []
            print(f"[INFO] Caching {len(self.files)} TKR images into RAM for '{split}' set...")
        
            for img_path, mask_path in tqdm(self.files, desc=f"Loading TKR-{split}"):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"❌ [Error] Skip bad image: {img_path}")
                    continue
                if mask is None:
                    print(f"❌ [Error] Skip bad mask: {mask_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = (mask > 0).astype(np.uint8) * 255

                self.cached_images.append(img)
                self.cached_masks.append(mask)
                valid_files.append((img_path, mask_path))
            
            self.files = valid_files
            print(f"[INFO] Final valid TKR samples for '{split}': {len(self.files)}")
    
    
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index):
        if self.cache_data and len(self.cached_images) == len(self.files):
            img = self.cached_images[index]
            mask = self.cached_masks[index]
        else:
            img_path, mask_path = self.files[index]
            
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"❌ 無法讀取圖片: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"❌ 無法讀取 mask: {mask_path}")
            mask = (mask > 0).astype(np.uint8) * 255 # 把 mask 變成我們看得到的
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
        else:
            mask = torch.from_numpy((mask > 0).astype(np.float32))
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return img, mask