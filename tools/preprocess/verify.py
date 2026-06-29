import argparse
import os
import cv2
import numpy as np
import glob
import random

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


def discover_datasets(processed_root):
    split_root = os.path.join(processed_root, "splits")
    if not os.path.exists(split_root):
        return []
    return sorted(
        name
        for name in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, name))
    )


def verify_dataset(processed_root, output_root, datasets, samples_num, seed):

    random.seed(seed)
    os.makedirs(output_root, exist_ok=True)
    
    if not os.path.exists(processed_root):
        print(f"❌ 找不到 {processed_root}，請先執行前處理！")
        return
    
    if not datasets:
        datasets = discover_datasets(processed_root)
    splits = ["train", "val"]
    
    for ds in datasets:
        for split in splits:
            print(f"🔍 Checking {ds} - {split} ...")
            
            img_dir = os.path.join(processed_root, ds, split, "images")
            mask_dir = os.path.join(processed_root, ds, split, "masks")
            
            out_dir = os.path.join(output_root, ds, split)
            os.makedirs(out_dir, exist_ok=True)
            
            all_images = glob.glob(os.path.join(img_dir, "*.png"))
            if not all_images:
                print(f"   ⚠️ No images found in {split}")
                continue
            
            # 隨機抽取 N 張
            sample_images = random.sample(all_images, min(len(all_images), samples_num))
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
    
    print(f"\n✅ 檢查完成！請去打開資料夾查看圖片：\n   📂 {output_root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create image/mask/overlay sanity checks.")
    parser.add_argument("--processed-root", default="data/processed/wound_clean")
    parser.add_argument("--output-root", default="outputs/sanity_check")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_dataset(
        processed_root=args.processed_root,
        output_root=args.output_root,
        datasets=args.datasets,
        samples_num=args.samples,
        seed=args.seed,
    )
