import os
import random

# ==========================================
# 設定（你可以改這裡）
# ==========================================
random.seed(42)  # 🔥 保證每次抽一樣

SPLIT_ROOT = "data/splits"

TARGET_NUM = {
    "DFU": 50,
    "Chronic": 50,
    "Abrasion": None,  # None = 全保留
    "Cut": None,
    "Laceration": None,
}


# ==========================================
# 核心 function
# ==========================================
def reduce_split(dataset_name, num_keep):
    split_path = os.path.join(SPLIT_ROOT, dataset_name, "train.txt")

    if not os.path.exists(split_path):
        print(f"❌ {split_path} not found")
        return

    with open(split_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    original_len = len(lines)

    # 🔥 如果不用減（None 或本來就小於）
    if num_keep is None or original_len <= num_keep:
        print(f"✔ {dataset_name}: keep all ({original_len})")
        return

    # 🔥 隨機抽樣
    selected = random.sample(lines, num_keep)

    # 🔥 覆蓋寫回
    with open(split_path, "w") as f:
        f.write("\n".join(selected))

    print(f"✅ {dataset_name}: {original_len} → {num_keep}")


# ==========================================
# 主程式
# ==========================================
def main():
    print("🚀 Creating finetune split...\n")

    for ds, num in TARGET_NUM.items():
        reduce_split(ds, num)

    print("\n🔥 Done!")


if __name__ == "__main__":
    main()