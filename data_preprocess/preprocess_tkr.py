import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# =========================
# 1. 路徑設定
# =========================
RAW_ROOT = "data_raw/TKR"
OUT_ROOT = "data/TKR"
MAPPING_FILE = os.path.join(RAW_ROOT, "tkr_mapping.xlsx")
TARGET_SIZE = (512, 512)

# =========================
# 2. Patient-level split
#    先用你前 20 張（4 個病患）的情況示範
# =========================
TRAIN_PATIENTS = ["7", "2", "3", "4", "5", "6"]
VAL_PATIENTS = ["1", "8"]
TEST_PATIENTS = []


def letterbox_resize(img, size, is_mask=False):
    """
    保持比例縮放，再補黑邊到指定尺寸
    size = (width, height)
    """
    img_h, img_w = img.shape[:2]
    w, h = size

    if (img_h == h) and (img_w == w):
        return img

    scale = min(w / img_w, h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if len(img.shape) == 3:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        new_img = np.zeros((h, w), dtype=np.uint8)

    dx = (w - new_w) // 2
    dy = (h - new_h) // 2

    if len(img.shape) == 3:
        new_img[dy:dy + new_h, dx:dx + new_w, :] = resized_img
    else:
        new_img[dy:dy + new_h, dx:dx + new_w] = resized_img

    return new_img


def build_mapping_from_excel_sheets(mapping_file):
    """
    讀取單一 Excel 的所有工作表
    適用你現在的 Excel 結構：
    - 欄名在第 2 列 => header=1
    - 需要欄位: 照片編號、病患編號
    - 若有兩個病患編號，取第一個（真正病患 ID）
    """
    excel_data = pd.read_excel(mapping_file, sheet_name=None, header=1)

    mapping = {}

    for sheet_name, df in excel_data.items():
        # 去掉欄名空白
        df.columns = [str(col).strip() for col in df.columns]

        # 找出所有叫「病患編號」的欄
        patient_cols = [col for col in df.columns if col == "病患編號"]

        if "照片編號" not in df.columns or len(patient_cols) == 0:
            print(f"⚠️ Skip sheet '{sheet_name}' because required columns not found.")
            print(f"   Columns: {list(df.columns)}")
            continue

        # 取第一個病患編號欄
        patient_col = patient_cols[0]

        for _, row in df.iterrows():
            photo_range = str(row["照片編號"]).strip()
            patient_id = str(row[patient_col]).strip()

            # 跳過空值 / 範例列
            if photo_range.lower() == "nan" or patient_id.lower() == "nan":
                continue
            if photo_range == "" or patient_id == "":
                continue
            if patient_id == "範例":
                continue

            try:
                # 範圍：例如 4031-4036
                if "-" in photo_range:
                    start_str, end_str = photo_range.split("-")
                    start_num = int(start_str)
                    end_num = int(end_str)

                    for num in range(start_num, end_num + 1):
                        mapping[f"IMG_{num}.jpg"] = patient_id
                        mapping[f"IMG_{num}.jpeg"] = patient_id
                        mapping[f"IMG_{num}.png"] = patient_id
                        mapping[f"IMG_{num}.bmp"] = patient_id

                # 單張：例如 4042
                else:
                    num = int(photo_range)
                    mapping[f"IMG_{num}.jpg"] = patient_id
                    mapping[f"IMG_{num}.jpeg"] = patient_id
                    mapping[f"IMG_{num}.png"] = patient_id
                    mapping[f"IMG_{num}.bmp"] = patient_id

            except ValueError:
                print(f"⚠️ Skip invalid photo_range in sheet '{sheet_name}': {photo_range}")
                continue

    return mapping


def get_split_from_patient(patient_id: str):
    if patient_id in TRAIN_PATIENTS:
        return "train"
    if patient_id in VAL_PATIENTS:
        return "val"
    if patient_id in TEST_PATIENTS:
        return "test"
    return None


def ensure_dirs():
    if os.path.exists(OUT_ROOT):
        print(f"🗑️ Cleaning old processed TKR directory: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, split, "masks"), exist_ok=True)

    os.makedirs(os.path.join(OUT_ROOT, "splits"), exist_ok=True)


def main():
    ensure_dirs()

    image_dir = os.path.join(RAW_ROOT, "images")
    mask_dir = os.path.join(RAW_ROOT, "masks")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

    mapping_dict = build_mapping_from_excel_sheets(MAPPING_FILE)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {image_dir}")

    split_names = defaultdict(list)

    print(f"[INFO] RAW_ROOT: {RAW_ROOT}")
    print(f"[INFO] OUT_ROOT: {OUT_ROOT}")
    print(f"[INFO] MAPPING_FILE: {MAPPING_FILE}")
    print(f"[INFO] Found {len(image_files)} raw images")
    print(f"[INFO] Found {len(mapping_dict)} filename-to-patient mappings")

    for fname in tqdm(image_files, desc="Processing TKR"):
        if fname not in mapping_dict:
            print(f"⚠️ No patient mapping for {fname}, skip")
            continue

        patient_id = mapping_dict[fname]
        split = get_split_from_patient(patient_id)

        if split is None:
            print(f"⚠️ Patient {patient_id} not assigned to split, skip: {fname}")
            continue

        img_path = os.path.join(image_dir, fname)
        stem, _ = os.path.splitext(fname)
        mask_path = os.path.join(mask_dir, f"{stem}.png")

        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found, skip: {fname}")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Bad image: {img_path}")
            continue
        if mask is None:
            print(f"❌ Bad mask: {mask_path}")
            continue

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # binary mask: 0 / 255
        mask = (mask > 0).astype(np.uint8) * 255

        # letterbox resize
        img_lb = letterbox_resize(img, TARGET_SIZE, is_mask=False)
        mask_lb = letterbox_resize(mask, TARGET_SIZE, is_mask=True)

        # processed 統一存 png
        out_name = f"{stem}.png"
        out_img_path = os.path.join(OUT_ROOT, split, "images", out_name)
        out_mask_path = os.path.join(OUT_ROOT, split, "masks", out_name)

        cv2.imwrite(out_img_path, cv2.cvtColor(img_lb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask_path, mask_lb)

        split_names[split].append(out_name)

    for split in ["train", "val", "test"]:
        txt_path = os.path.join(OUT_ROOT, "splits", f"{split}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(split_names[split]))

        print(f"✅ {split}: {len(split_names[split])} samples")

    print("\n🎉 TKR preprocess finished!")


if __name__ == "__main__":
    main()