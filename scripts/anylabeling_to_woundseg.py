"""
把 AnyLabeling 輸出的 polygon JSON 轉成 woundseg 訓練用格式。

輸入目錄結構（AnyLabeling 標完的樣子）：
    src/
    ├── IMG_001.jpg
    ├── IMG_001.json          # AnyLabeling 標註
    ├── IMG_002.jpg
    ├── IMG_002.json
    └── ...

輸出（合併進你既有 data/processed/wound/<class>/<split>/）：
    dst/<class>/<split>/
    ├── images/IMG_001.jpg
    └── masks/IMG_001.png     # 0 / 255 binary mask

用法：
    python scripts/anylabeling_to_woundseg.py \
        --src data/raw/sam_collected/Cut \
        --class-name Cut \
        --split train \
        --dst data/processed/wound

備註：
    - 只取 label 與 --class-name 相同的 polygon（不分大小寫）
    - 同一張圖多 polygon 會 union 合成一張 mask
    - 邊界做一次輕微 morphology close + open 平滑
    - 若 --class-name 為 "_any"，任意 label 都接受（適合公開資料只有單類）
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

VALID_CLASSES = {
    "cut", "abrasion", "laceration", "footulcer", "chronic",
    "Cut", "Abrasion", "Laceration", "DFU", "Chronic",
}
VALID_SPLITS = {"train", "val", "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def polygon_to_mask(shapes: list[dict], h: int, w: int, target_label: str) -> np.ndarray:
    """把符合 target_label 的 polygon 渲染成 binary mask (uint8, 0/255)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    accept_any = target_label == "_any"
    for shape in shapes:
        if not accept_any and shape.get("label", "").lower() != target_label.lower():
            continue
        if shape.get("shape_type") != "polygon":
            # 也接受 mask / rectangle，但這裡只處理 polygon（AnyLabeling+SAM 預設）
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 255)

    # 平滑邊界
    if mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_image(json_path: Path) -> Path | None:
    """找 json 對應的圖檔。優先用 imagePath 欄位，找不到就用 stem 比對。"""
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    image_path_field = meta.get("imagePath")
    if image_path_field:
        candidate = (json_path.parent / image_path_field).resolve()
        if candidate.exists():
            return candidate
    # fallback：同 stem 比對
    for ext in IMG_EXTS:
        cand = json_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def process(args: argparse.Namespace) -> dict:
    src = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    dst_img = dst_root / args.class_name / args.split / "images"
    dst_msk = dst_root / args.class_name / args.split / "masks"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_msk.mkdir(parents=True, exist_ok=True)

    stats = {"json_total": 0, "ok": 0, "no_image": 0, "no_polygon": 0, "empty_mask": 0}

    for json_path in sorted(src.glob("*.json")):
        stats["json_total"] += 1
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = meta.get("shapes", [])
        if not shapes:
            stats["no_polygon"] += 1
            continue

        img_path = find_image(json_path)
        if img_path is None:
            print(f"[skip] no image for {json_path.name}")
            stats["no_image"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cv2 cannot read {img_path.name}")
            stats["no_image"] += 1
            continue
        h, w = img.shape[:2]

        # 若 JSON 裡有 imageHeight/Width 而和實際圖不同，以實際為準
        mask = polygon_to_mask(shapes, h, w, args.class_name)
        if mask.sum() == 0:
            print(f"[skip] no polygon matched class '{args.class_name}' in {json_path.name}")
            stats["empty_mask"] += 1
            continue

        # 寫出
        out_img = dst_img / img_path.name
        out_msk = dst_msk / (img_path.stem + ".png")

        if args.copy_images:
            shutil.copy2(img_path, out_img)
        else:
            # 預設用 symlink 省空間（同碟才會成功；失敗就 fallback copy）
            try:
                if out_img.exists() or out_img.is_symlink():
                    out_img.unlink()
                out_img.symlink_to(img_path)
            except OSError:
                shutil.copy2(img_path, out_img)

        cv2.imwrite(str(out_msk), mask)
        stats["ok"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="AnyLabeling 標完的目錄（含 jpg + json）")
    parser.add_argument("--class-name", required=True,
                        help=f"要產生的類別名稱：{VALID_CLASSES} 或 '_any'")
    parser.add_argument("--split", default="train", choices=sorted(VALID_SPLITS))
    parser.add_argument("--dst", default="data/processed/wound",
                        help="輸出根目錄（預設 data/processed/wound）")
    parser.add_argument("--copy-images", action="store_true",
                        help="預設用 symlink 省空間，加此 flag 改實體複製")
    args = parser.parse_args()

    if args.class_name != "_any" and args.class_name not in VALID_CLASSES:
        parser.error(f"--class-name 必須是 {VALID_CLASSES} 或 '_any'")

    stats = process(args)
    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"輸出位置：{Path(args.dst).resolve() / args.class_name / args.split}")


if __name__ == "__main__":
    main()
