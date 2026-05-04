import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

def rename_images(folder_path, prefix="cut", start_idx=1):
    files = sorted(os.listdir(folder_path))  # 排序很重要！

    idx = start_idx

    for file in files:
        old_path = os.path.join(folder_path, file)

        # 跳過資料夾
        if not os.path.isfile(old_path):
            continue

        # 只處理圖片
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        new_name = f"{prefix}_{idx:03d}.png"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"{file} → {new_name}")

        idx += 1


# 使用
rename_images("data_raw/Laceration")