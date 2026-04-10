import os
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ==========================================
# 設定
# ==========================================
RAW_ROOT = "data_raw"
OUT_ROOT = "data"
TARGET_SIZE = (512, 512)

DATASET_MAP = {
    "WoundSeg": "DFU",
    "FootUlcer": "DFU",
    "CO2Wound": "Chronic",
    "Abrasion": "Abrasion",
    # 之後可加
    # "Cut": "Cut",
    # "Laceration": "Laceration"
}


# ==========================================
# Letterbox Resize
# ==========================================
def letterbox_resize(img, size, is_mask=False):
    h, w = size
    ih, iw = img.shape[:2]

    if ih == h and iw == w:
        return img

    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    if len(img.shape) == 3:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((h, w), dtype=np.uint8)

    dx = (w - nw) // 2
    dy = (h - nh) // 2

    if len(img.shape) == 3:
        canvas[dy:dy+nh, dx:dx+nw] = resized
    else:
        canvas[dy:dy+nh, dx:dx+nw] = resized

    return canvas


# ==========================================
# 處理 dataset
# ==========================================
def process_dataset(img_dir, mask_dir):

    img_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_paths += glob.glob(os.path.join(img_dir, ext))

    img_paths = sorted(img_paths)
    data = []

    for img_path in tqdm(img_paths):
        fname = os.path.basename(img_path)
        basename = os.path.splitext(fname)[0]

        # 找 mask
        mask_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(mask_dir, basename + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            continue

        # 讀 image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 讀 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # binary mask
        mask = mask.astype(np.float32)
        if mask.max() > 1:
            mask /= 255.0

        mask[mask >= 0.5] = 255
        mask[mask < 0.5] = 0
        mask = mask.astype(np.uint8)

        # resize
        img = letterbox_resize(img, TARGET_SIZE)
        mask = letterbox_resize(mask, TARGET_SIZE, True)

        data.append({
            "name": fname,   # 🔥 保留原始檔名
            "img": img,
            "mask": mask
        })

    return data


# ==========================================
# 存檔
# ==========================================
def save_data(data_list, out_dir):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    names = []

    for item in data_list:
        name = item["name"]

        cv2.imwrite(
            os.path.join(out_dir, "images", name),
            cv2.cvtColor(item["img"], cv2.COLOR_RGB2BGR)
        )

        cv2.imwrite(
            os.path.join(out_dir, "masks", name),
            item["mask"]
        )

        names.append(name)

    return names


# ==========================================
# 主程式
# ==========================================
def main():

    # ⚠️ 清空舊資料（小心）
    if os.path.exists(OUT_ROOT):
        print(f"🗑️ Removing old {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    for raw_name, new_name in DATASET_MAP.items():

        print(f"\n🚀 {raw_name} → {new_name}")

        img_dir = os.path.join(RAW_ROOT, raw_name, "images")
        mask_dir = os.path.join(RAW_ROOT, raw_name, "masks")

        if not os.path.exists(img_dir):
            print("❌ Skip (no folder)")
            continue

        data = process_dataset(img_dir, mask_dir)

        if len(data) == 0:
            print("❌ No valid data")
            continue

        # split (固定 random seed 很重要)
        train, val = train_test_split(data, test_size=0.2, random_state=42)

        out_base = os.path.join(OUT_ROOT, new_name)

        t_names = save_data(train, os.path.join(out_base, "train"))
        v_names = save_data(val, os.path.join(out_base, "val"))

        print(f"✅ Train: {len(t_names)} | Val: {len(v_names)}")


if __name__ == "__main__":
    main()