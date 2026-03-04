import os
import re
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


RAW_ROOT = "data/raw"
LABELED_ROOT = f"{RAW_ROOT}/labeled"
TEST_ROOT = f"{RAW_ROOT}/test"
OUT_ROOT = "data/processed"
TARGET_SIZE = (512, 512)
PREFIX_MAP = {
    "WoundSeg": "WS",
    "CO2Wound": "CO2",
    "FootUlcer": "FU"
}


def letterbox_resize(img, size, is_mask=False):

    # 1. 取得原始圖片的 高(ih) 與 寬(iw)
    img_h, img_w = img.shape[:2]
    w, h = size
    
    # if the size match exactly, return the original image directly.
    if (int(img_h) == int(h)) and (int(img_w) == int(w)):
        return img
    
    # 2. 計算縮放比例 (Scale)
    scale = min(w/img_w, h/img_h)
    
    # 3. 計算縮放後的新尺寸 (nw, nh)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # 4. 決定縮放演算法 (Interpolation)
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        
    # 5. 執行縮放 (這時候還沒有黑邊，只是把圖變小了)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # 6. 建立畫布 (Canvas)
    # 產生一張全黑 (0) 的 512x512 底圖
    if len(img.shape) == 3:
        # 彩色圖: (512, 512, 3)
        new_img = np.zeros((h, w, 3), np.uint8)
    else:
        # 灰階圖/Mask: (512, 512)
        new_img = np.zeros((h, w), dtype=np.uint8)
    
    # 7. 計算貼圖位置 (Centering)
    # 我們要把縮小後的圖「置中」貼在黑畫布上
    dx = (w-new_w) // 2
    dy = (h-new_h) // 2
    
    # 8. 貼上去 (Paste)
    if len(img.shape) == 3:
        new_img[dy:dy+new_h, dx:dx+new_w, :] = resized_img
    else:
        new_img[dy:dy+new_h, dx:dx+new_w] = resized_img
        
    return new_img


def generate_new_name(original_fname, dataset_name):
    """
    輸入: IMG435.png, CO2Wound
    輸出: CO2_435.png
    """
    base_name, _ = os.path.splitext(original_fname)
    prefix = PREFIX_MAP.get(dataset_name, dataset_name)
    
    # 使用 Regex 找出檔名中的所有數字序列
    numbers = re.findall(r'\d+', base_name)
    
    if numbers and dataset_name != "FootUlcer":
        # 通常 ID 是檔名中最後一組數字 (避免抓到日期)
        # 例如: IMG_20251212_005.jpg -> 取 005
        real_id = numbers[-1]
    else:
        # 如果檔名完全沒數字 (例如 image.png)，就保留原名
        real_id = base_name
        
    return f"{prefix}_{real_id}.png"


def process_folder(dataset_name, subset_type, input_img_dir, input_mask_dir):
    
    if not os.path.exists(input_img_dir):
        print(f"   ⚠️ Path not found: {input_img_dir}")
        return []
    
    # 1. 搜尋圖片
    extension = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in extension:
        img_paths.extend(glob.glob(os.path.join(input_img_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(input_img_dir, ext.upper())))

    img_paths = sorted(list(set(img_paths))) # 去除重複並排序
    processed_data = []
    
    if not img_paths:
        print(f"   ⚠️ No images found in {dataset_name}/{subset_type}")
        return []
    
    for img_path in tqdm(img_paths, desc=f"Processing: {dataset_name}"):
        fname = os.path.basename(img_path)
        basename, _ = os.path.splitext(fname)

        candidate = [
            os.path.join(input_mask_dir, fname),                  # 1. 完全同名 (image.jpg 對 image.jpg)
            os.path.join(input_mask_dir, basename + ".png"),      # 2. 同名但副檔名是 png (常見：圖片有壓縮，Mask 無壓縮)
            os.path.join(input_mask_dir, basename + ".jpg")       # 3. 同名但副檔名是 jpg
        ]
        
        mask_path = None
        for c in candidate:
            if os.path.exists(c):
                mask_path = c
                break
        
        if subset_type == 'labeled' and mask_path is None:
            continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 不管原本 Mask 是 RGB 彩色還是灰階，這裡通通變成單層灰階。
        # 不管原本 Mask 是 0/1 (全黑) 還是 0/255 (黑白)，這裡通通變成 0/255。
        # 結果：所有的 Mask 變成了統一規格，且肉眼可見（白色的傷口）
        mask = None
        if mask_path is not None:
            mask_raw = cv2.imread(mask_path)
            
            if mask_raw is not None:
                if len(mask_raw.shape) == 3:
                    mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
                else:
                    mask = mask_raw
                
                mask = mask.astype(np.float32)
                if mask.max() > 1.0:
                    mask /= 255.0
                
                mask[mask >= 0.5] = 255
                mask[mask < 0.5] = 0
                mask = mask.astype(np.uint8)
        
        img_lb = letterbox_resize(img, TARGET_SIZE)
        if mask is not None:
            mask_lb = letterbox_resize(mask, TARGET_SIZE, True)
        else:
            # 如果是 Test set 且沒有 Mask，自動生成一張全黑圖當作「替身」
            # 這樣程式才不會因為變數是 None 而報錯
            mask_lb = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
        
        new_filename = generate_new_name(fname, dataset_name)
        processed_data.append({
            "name": new_filename,
            "img": img_lb,
            "mask": mask_lb
        })
    
    return processed_data

def save_data(data_list, out_dir):
    
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    
    names = []
    for item in data_list:
        name = item['name']
        cv2.imwrite(os.path.join(out_dir, "images", name), cv2.cvtColor(item['img'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "masks", name), item['mask'])
        names.append(name)
    return names


def main():
    
    if os.path.exists(OUT_ROOT):
        print(f"🗑️  Cleaning up old data at: {OUT_ROOT} ...")
        shutil.rmtree(OUT_ROOT, ignore_errors=True) # 遞迴刪除整個資料夾
    
    datasets = ["WoundSeg", "CO2Wound", "FootUlcer"]
    print(f"[INFO] Raw Root: {RAW_ROOT}")
    print(f"[INFO] Labeled Root: {LABELED_ROOT}")
    print(f"[INFO] Test Root: {TEST_ROOT}")
    
    for ds_name in datasets:
        print(f"\n🚀 Pipeline: {ds_name} (Prefix: {PREFIX_MAP.get(ds_name)})")
        
        labeled_img = os.path.join(LABELED_ROOT, ds_name, "images")
        labeled_mask = os.path.join(LABELED_ROOT, ds_name, "masks")
        test_img = os.path.join(TEST_ROOT, ds_name, "images")
        test_mask = os.path.join(TEST_ROOT, ds_name, "masks")
        
        # 成品區
        out_base = os.path.join(OUT_ROOT, ds_name)
        os.makedirs(out_base, exist_ok=True)
        
        # 紀錄區 (Splits - 存放 txt 的地方)
        split_dir = os.path.join(OUT_ROOT, "splits", ds_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # 1. 呼叫生產線 (process_folder)
        # 這時候，所有的圖片都已經在記憶體變成 512x512 且改好名字了
        print("  -> Processing Labeled set...")
        data = process_folder(ds_name, "labeled", labeled_img, labeled_mask)
        
        if len(data) > 0:
            # 2. 黃金比例切分 (Train/Val Split)
            #run4 改成 Train/Val 8:2
            # test_size=0.2 代表切出 20% 給驗證集 (Val)，剩下 80% 給訓練集 (Train)
            # random_state=42 確保每次切出來的結果都一樣
            train, val = train_test_split(data, test_size=0.2, random_state=42)
            
            # 3. 實際存檔 (把記憶體寫入硬碟)
            # 這時候才會產生 data/processed/WoundSeg/train/images/WS_001.png
            t_names = save_data(train, os.path.join(out_base, "train"))
            v_names = save_data(val, os.path.join(out_base, "val"))
            
            # 4. 寫入點名簿 (.txt)
            # 這些 txt 檔案就是以後 Dataset Loader 讀取的依據
            with open(os.path.join(split_dir, "train.txt"), "w") as f: 
                f.write("\n".join(t_names))
            with open(os.path.join(split_dir, "val.txt"), "w") as f: 
                f.write("\n".join(v_names))
                
            print(f"     ✅ Train: {len(t_names)} | Val: {len(v_names)}")
        
        # 1. 呼叫生產線 (process_folder)
        print("  -> Processing Test set...")
        t_data = process_folder(ds_name, "test", test_img, test_mask)
        
        if len(t_data) > 0:
            # 2. 測試集不需要切分，直接全存！
            test_names = save_data(t_data, os.path.join(out_base, "test"))
            
            # 3. 寫入 test.txt
            with open(os.path.join(split_dir, "test.txt"), "w") as f: 
                f.write("\n".join(test_names))
                
            print(f"     ✅ Test: {len(test_names)}")


if __name__ == "__main__":
    main()