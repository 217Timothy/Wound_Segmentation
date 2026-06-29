#!/usr/bin/env python3
"""把資料夾裡的圖片重新命名成 '<prefix>_<index>.<ext>'（依排序順序）。

改良點（相對於舊版）：
    1. 保留原副檔名（.jpg 維持 .jpg）。不強制轉 .png。
       若真的要轉檔，加 --to-ext png 會用 PIL 真的轉檔（不只改名）。
    2. 自動依檔案數計算填零寬度（避免 999 → 1000 排序壞掉）。
    3. --dry-run：先預覽 mapping，不動檔案。
    4. 產 rename_mapping.json 對照表（可逆，可用 --undo 還原）。
    5. 預先檢查目標檔名衝突，避免覆蓋。
    6. --with-masks：偵測同層 ../masks/ 並同步改 mask 檔名（mask 副檔名固定 .png）。

範例：
    # 預覽（強烈建議先跑這個）
    python tools/rename_images.py --dir data/raw/multiclass/cut --prefix cut --dry-run

    # 實際 rename
    python tools/rename_images.py --dir data/raw/multiclass/cut --prefix cut

    # segmentation/ 下要同步改 mask
    python tools/rename_images.py \\
        --dir data/raw/segmentation/cut/images \\
        --prefix cut --with-masks

    # 還原上一次 rename
    python tools/rename_images.py --dir data/raw/multiclass/cut --undo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAPPING_FILENAME = "rename_mapping.json"


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def auto_pad(n: int) -> int:
    """依檔案數自動算填零位數（至少 3 位，>999 變 4 位，>9999 變 5 位）。"""
    if n <= 999:
        return 3
    if n <= 9999:
        return 4
    return 5


def plan_renames(files: list[Path], prefix: str, start_idx: int,
                 to_ext: str | None, pad: int | None) -> list[tuple[Path, Path]]:
    """產生 (舊路徑, 新路徑) 列表"""
    width = pad if pad is not None else auto_pad(len(files) + start_idx - 1)
    plan = []
    for i, src in enumerate(files):
        idx = start_idx + i
        ext = ("." + to_ext.lower()) if to_ext else src.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"  # 統一短副檔名
        dst = src.parent / f"{prefix}_{idx:0{width}d}{ext}"
        plan.append((src, dst))
    return plan


def check_conflicts(plan: list[tuple[Path, Path]]) -> list[str]:
    """檢查兩種衝突：
       (a) 兩個 src rename 後撞同名（理論上不會）
       (b) 新檔名跟「不在 plan 裡」的既有檔案撞名
    """
    errors = []
    dsts = [dst for _, dst in plan]
    if len(set(dsts)) != len(dsts):
        errors.append("目標檔名有重複 → 程式邏輯錯誤")

    src_set = {src for src, _ in plan}
    for src, dst in plan:
        if dst != src and dst.exists() and dst not in src_set:
            errors.append(f"目標 {dst.name} 已存在且不在 rename 計畫內")
    return errors


def safe_rename_two_phase(plan: list[tuple[Path, Path]]) -> None:
    """兩階段 rename，避開「dst 是某個 src 的舊名」造成的循環覆蓋。
       第一階段：全部 src → tmp 名稱
       第二階段：tmp → dst
    """
    tmp_plan = []
    for i, (src, dst) in enumerate(plan):
        if src == dst:
            tmp_plan.append((src, src))
            continue
        tmp = src.parent / f"._rename_tmp_{i}_{src.name}"
        os.rename(src, tmp)
        tmp_plan.append((tmp, dst))
    for tmp, dst in tmp_plan:
        if tmp == dst:
            continue
        os.rename(tmp, dst)


def convert_format(plan: list[tuple[Path, Path]], target_ext: str) -> None:
    """真的把圖讀進來再以新格式存檔（用在 --to-ext 時）"""
    from PIL import Image
    for src, dst in plan:
        with Image.open(src) as im:
            im = im.convert("RGB") if target_ext.lower() in ("jpg", "jpeg") else im
            im.save(dst)
        if src.exists() and src != dst:
            src.unlink()


def write_mapping(folder: Path, plan: list[tuple[Path, Path]]) -> Path:
    """寫對照表，供 --undo 還原。"""
    mp = folder / MAPPING_FILENAME
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folder": str(folder),
        "mapping": [{"old": src.name, "new": dst.name} for src, dst in plan if src.name != dst.name],
    }
    mp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return mp


def find_mask_for(image_path: Path, masks_dir: Path) -> Path | None:
    """在 masks_dir 找對應 mask（同 stem，可能副檔名 .png/.jpg）。"""
    for ext in IMAGE_EXTS:
        cand = masks_dir / (image_path.stem + ext)
        if cand.exists():
            return cand
    return None


def do_rename(args: argparse.Namespace) -> None:
    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        sys.exit(f"[err] 找不到資料夾：{folder}")

    files = list_images(folder)
    if not files:
        sys.exit(f"[err] {folder} 沒有任何圖片")

    plan = plan_renames(files, args.prefix, args.start, args.to_ext, args.pad)

    # 衝突檢查
    errs = check_conflicts(plan)
    if errs:
        for e in errs:
            print(f"[err] {e}")
        sys.exit(1)

    # 同步 mask（若 --with-masks）
    mask_plan: list[tuple[Path, Path]] = []
    if args.with_masks:
        masks_dir = folder.parent / "masks"
        if not masks_dir.is_dir():
            print(f"[warn] --with-masks 但找不到 {masks_dir}，跳過 mask 同步")
        else:
            for (src, dst), orig in zip(plan, files):
                m = find_mask_for(orig, masks_dir)
                if m is None:
                    print(f"[warn] 圖 {orig.name} 找不到對應 mask")
                    continue
                # mask 副檔名固定 .png
                mdst = masks_dir / (dst.stem + ".png")
                mask_plan.append((m, mdst))

    # 預覽
    print(f"\n=== {'DRY-RUN' if args.dry_run else 'EXECUTE'} ===")
    print(f"資料夾：{folder}")
    print(f"原圖數：{len(files)}")
    print(f"填零寬度：{args.pad or auto_pad(len(files) + args.start - 1)}")
    print(f"前 5 筆預覽：")
    for src, dst in plan[:5]:
        marker = "✓" if src.name != dst.name else "·"
        print(f"  {marker} {src.name}  →  {dst.name}")
    if len(plan) > 5:
        print(f"  ... 共 {len(plan)} 筆")

    if mask_plan:
        print(f"\n同步 mask：{len(mask_plan)} 個")
        for src, dst in mask_plan[:3]:
            print(f"  {src.name}  →  {dst.name}")
        if len(mask_plan) > 3:
            print(f"  ... 共 {len(mask_plan)} 個")

    if args.dry_run:
        print("\n[dry-run] 沒有實際更動。確認沒問題後拿掉 --dry-run 重跑。")
        return

    # 真的執行
    if args.to_ext:
        print(f"\n[info] --to-ext 模式：真的轉檔成 .{args.to_ext}（會慢一點）")
        convert_format(plan, args.to_ext)
        if mask_plan:
            convert_format(mask_plan, "png")  # mask 強制 PNG
    else:
        safe_rename_two_phase(plan)
        if mask_plan:
            safe_rename_two_phase(mask_plan)

    mp = write_mapping(folder, plan)
    print(f"\n[ok] 重命名完成 {len(plan)} 個檔案")
    print(f"[ok] 對照表寫入 {mp}（用 --undo 可還原）")


def do_undo(args: argparse.Namespace) -> None:
    folder = Path(args.dir).resolve()
    mp = folder / MAPPING_FILENAME
    if not mp.exists():
        sys.exit(f"[err] 找不到 {mp}，沒有可還原的紀錄")

    data = json.loads(mp.read_text())
    pairs = data["mapping"]
    # 反向：把新檔名改回舊檔名
    plan: list[tuple[Path, Path]] = []
    for item in pairs:
        new_p = folder / item["new"]
        old_p = folder / item["old"]
        if new_p.exists():
            plan.append((new_p, old_p))
        else:
            print(f"[warn] {new_p.name} 不存在，跳過")

    if not plan:
        sys.exit("[err] 沒有可還原的檔案")

    errs = check_conflicts(plan)
    if errs:
        for e in errs:
            print(f"[err] {e}")
        sys.exit(1)

    print(f"=== UNDO ===")
    print(f"還原 {len(plan)} 個檔案")
    for src, dst in plan[:5]:
        print(f"  {src.name}  →  {dst.name}")
    if len(plan) > 5:
        print(f"  ... 共 {len(plan)} 筆")

    if args.dry_run:
        print("\n[dry-run] 沒有實際更動")
        return

    safe_rename_two_phase(plan)
    mp.unlink()
    print(f"[ok] 還原完成，移除 {mp.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="要 rename 的資料夾")
    parser.add_argument("--prefix", help="檔名前綴（rename 必填）")
    parser.add_argument("--start", type=int, default=1, help="起始編號（預設 1）")
    parser.add_argument("--pad", type=int, default=None,
                        help="填零位數，預設依張數自動算（避免 999→1000 排序壞）")
    parser.add_argument("--to-ext", choices=["jpg", "png"], default=None,
                        help="真的轉檔成此格式（會用 PIL 重存）；預設保留原副檔名")
    parser.add_argument("--with-masks", action="store_true",
                        help="若 --dir 是 images/，同步改 ../masks/ 對應檔名")
    parser.add_argument("--dry-run", action="store_true", help="只預覽不執行")
    parser.add_argument("--undo", action="store_true", help="還原上次 rename")
    args = parser.parse_args()

    if args.undo:
        do_undo(args)
    else:
        if not args.prefix:
            parser.error("--prefix 是必填（除非用 --undo）")
        do_rename(args)


if __name__ == "__main__":
    main()
