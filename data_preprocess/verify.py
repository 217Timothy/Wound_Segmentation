import os
import cv2
import numpy as np
import glob
import random
from tqdm import tqdm

# ===========================
# 設定區
# ===========================
PROCESSED_ROOT = "data/processed"   # 讀取處理好的資料
OUTPUT_ROOT = "results/sanity_check" # 輸出檢查圖的位置
os.makedirs(os.path.join(OUTPUT_ROOT), exist_ok=True)
SAMPLES_NUM = 8

def overlay_image(img, mask, color=(0, 255, 0), alpha=0.5):
    """
    將 Mask 疊加在圖片上
    color: 預設為綠色 (0, 255, 0) 因為傷口通常是紅的，綠色對比最明顯
    """
    
    # 轉成彩色以便疊加
    mask_color = np.zeros_like(img)
    mask_color[mask == 255] = color
    
    img_copy = img.copy()
    overlay = cv2.addWeighted(img_copy, 1, mask_color, alpha, 0)
    
    # 畫出輪廓 (讓邊緣更清楚)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2) # 黃色邊框
    
    return overlay


def verify_dataset():
    
    if not os.path.exists(PROCESSED_ROOT):
        print(f"❌ 找不到 {PROCESSED_ROOT}，請先執行前處理！")
        return
    
    datasets = ["WoundSeg", "CO2Wound", "FootUlcer"]
    splits = ["train", "val"]
    
    for ds in datasets:
        for split in splits:
            print(f"🔍 Checking {ds} - {split} ...")
            
            img_dir = os.path.join(PROCESSED_ROOT, ds, split, "images")
            mask_dir = os.path.join(PROCESSED_ROOT, ds, split, "masks")
            
            out_dir = os.path.join(OUTPUT_ROOT, ds, split)
            os.makedirs(out_dir, exist_ok=True)
            
            all_images = glob.glob(os.path.join(img_dir, "*.png"))
            if not all_images:
                print(f"   ⚠️ No images found in {split}")
                continue
            
            # 隨機抽取 N 張
            sample_images = random.sample(all_images, min(len(all_images), SAMPLES_NUM))
            for img_path in sample_images:
                fname = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, fname)
                
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    print(f"   ❌ Error reading {fname}")
                    continue
                
                overlay = overlay_image(img, mask)
                
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                combine = np.hstack([img, mask_bgr, overlay])
                
                save_path = os.path.join(out_dir, f"check_{fname}")
                cv2.imwrite(save_path, combine)
    
    print(f"\n✅ 檢查完成！請去打開資料夾查看圖片：\n   📂 {OUTPUT_ROOT}")


if __name__ == "__main__":
    verify_dataset()