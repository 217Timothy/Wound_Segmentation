import os
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()


def convert_and_remove_heic(input_dir, jpg_quality=95):
    files = sorted(os.listdir(input_dir))

    heic_files = [
        f for f in files
        if f.lower().endswith(".heic")
    ]

    if not heic_files:
        print(f"[WARN] No HEIC files found in: {input_dir}")
        return

    print(f"[INFO] Found {len(heic_files)} HEIC files.")

    for fname in tqdm(heic_files, desc="Convert + Remove HEIC"):
        in_path = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(input_dir, f"{stem}.jpg")

        try:
            # 1️⃣ 先轉檔
            img = Image.open(in_path).convert("RGB")
            img.save(out_path, "JPEG", quality=jpg_quality)

            # 2️⃣ 確認成功才刪原檔
            if os.path.exists(out_path):
                os.remove(in_path)

        except Exception as e:
            print(f"[ERROR] Failed {fname}: {e}")

    print(f"\n✅ Done! HEIC → JPG completed & originals removed.")


if __name__ == "__main__":
    input_dir = "data/TKR/test/images"
    convert_and_remove_heic(input_dir)