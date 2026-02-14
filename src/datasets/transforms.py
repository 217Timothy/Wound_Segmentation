import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(512, 512)):
    return A.Compose([
        # 1. 隨機翻轉
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5), # 隨機旋轉 +/- 30度
        
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        
        A.CoarseDropout(
            num_holes_range=(1, 8),     # 對應原本的 min_holes=1, max_holes=8
            hole_height_range=(8, 32),  # 洞的高度範圍：8 到 32 像素
            hole_width_range=(8, 32),   # 洞的寬度範圍：8 到 32 像素
            fill=0,                     # 對應原本的 fill_value=0 (圖片挖洞填黑色)
            fill_mask=0,                # 對應原本的 mask_fill_value=0 (遮罩也填黑色)
            p=0.3
        ),
        
        # 2. 色彩變換 (只影響 Image，不影響 Mask)
        # 傷口顏色很重要，所以我們輕微調整就好
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=5, p=0.3),
        
        # 3. 確保尺寸 (雖然前處理做過了，但雙重保險是好習慣)
        A.Resize(height=img_size[1], width=img_size[0]),
        
        # 4. 正規化
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # 4. 轉 Tensor (這步會自動除以 255 並轉 CHW，如果在 Dataset 手寫了就不用這行)
        # 為了配合上面 Dataset 手寫的邏輯，這裡我先註解掉，讓 Dataset 自己處理 Tensor 轉換
        ToTensorV2(),
    ])


# 驗證集通常只做 Resize (確保尺寸對)，不做翻轉等破壞性增強
def get_val_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(height=img_size[1], width=img_size[0]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])