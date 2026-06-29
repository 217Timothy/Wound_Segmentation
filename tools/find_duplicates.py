"""
偵測資料集裡的重複圖片。

掃描三種重複：
    1. 完全相同（同一個檔案，MD5 hash 相同）
    2. 視覺上幾乎相同（perceptual hash，可抓 resize / 不同壓縮 / 輕微編輯）
    3. 跨類別 / 跨 split 衝突（更嚴重）

用法：
    # 掃 processed/wound（建議優先跑這個，看 train/val 有沒有重疊 → 資料洩漏）
    python tools/find_duplicates.py --root data/processed/wound

    # 掃 raw/multiclass
    python tools/find_duplicates.py --root data/raw/multiclass

    # 同時掃多個根目錄，看跨來源重複
    python tools/find_duplicates.py \
        --root data/raw/multiclass \
        --root data/raw/segmentation \
        --root data/processed/wound

輸出：
    duplicates_report.json   完整列表
    終端機印出摘要
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_of_image(path: Path) -> str | None:
    """perceptual hash：對 resize / 重新壓縮 / 輕微變動有抗性"""
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return str(imagehash.phash(im, hash_size=8))
    except Exception:
        return None


def iter_images(root: Path):
    """收所有圖片，跳過 masks/ 子目錄（mask 重複不重要）"""
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        # 跳過 mask 目錄
        if "mask" in p.parent.name.lower():
            continue
        yield p


def label_of(path: Path, roots: list[Path]) -> str:
    """用相對路徑當作 'label'，給人看是哪個類別/split"""
    for r in roots:
        try:
            return str(path.relative_to(r.parent))
        except ValueError:
            continue
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", required=True, help="可重複指定多個根目錄")
    parser.add_argument("--no-phash", action="store_true", help="不算 perceptual hash（更快但只能抓完全重複）")
    parser.add_argument("--out", default="duplicates_report.json")
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    for r in roots:
        if not r.exists():
            raise SystemExit(f"[err] 根目錄不存在：{r}")

    # 收集所有圖
    print("[..] 掃描檔案...")
    all_files = []
    for r in roots:
        files = list(iter_images(r))
        print(f"     {r.name}: {len(files)} 張")
        all_files.extend(files)
    print(f"[ok] 共 {len(all_files)} 張圖")

    # 算 MD5
    print("[..] 算 MD5 hash（完全相同檢測）...")
    md5_to_paths: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        try:
            md5_to_paths[md5_of_file(p)].append(p)
        except Exception as e:
            print(f"     skip {p}: {e}")

    exact_dups = {h: paths for h, paths in md5_to_paths.items() if len(paths) > 1}
    print(f"[ok] 完全相同 (MD5)：{len(exact_dups)} 組重複")

    # 算 perceptual hash
    near_dups: dict[str, list[Path]] = {}
    if not args.no_phash:
        try:
            from PIL import Image  # noqa: F401
            import imagehash  # noqa: F401
        except ImportError:
            print("[warn] 沒裝 imagehash，跳過視覺相似檢測。`pip install imagehash Pillow` 才能跑")
        else:
            print("[..] 算 perceptual hash（視覺相似檢測，較慢）...")
            phash_to_paths: dict[str, list[Path]] = defaultdict(list)
            for i, p in enumerate(all_files):
                if i % 200 == 0 and i > 0:
                    print(f"     ...{i}/{len(all_files)}")
                ph = phash_of_image(p)
                if ph is not None:
                    phash_to_paths[ph].append(p)
            near_dups = {h: paths for h, paths in phash_to_paths.items()
                         if len(paths) > 1 and h not in exact_dups}
            print(f"[ok] 視覺幾乎相同 (phash)：{len(near_dups)} 組")

    # 分析重複類型
    def classify(paths: list[Path]) -> str:
        labels = sorted({label_of(p, roots).split("/")[0] + "/" + label_of(p, roots).split("/")[1]
                         if len(label_of(p, roots).split("/")) > 1 else label_of(p, roots).split("/")[0]
                         for p in paths})
        # 比較粗略：看頂層資料夾是不是不一樣（跨類別 / 跨 split）
        top_labels = {label_of(p, roots).split("/")[1] if len(label_of(p, roots).split("/")) > 1 else "?"
                      for p in paths}
        if len(top_labels) > 1:
            return f"跨類別/跨來源 ⚠️ ({sorted(top_labels)})"
        return "同類別內"

    # 報告
    report = {
        "roots": [str(r) for r in roots],
        "total_images": len(all_files),
        "exact_duplicates": [],
        "near_duplicates": [],
    }

    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)

    if exact_dups:
        print(f"\n[完全相同 - {len(exact_dups)} 組]")
        cross_class_exact = 0
        for h, paths in list(exact_dups.items())[:10]:
            kind = classify(paths)
            if "跨" in kind:
                cross_class_exact += 1
            print(f"  {kind}")
            for p in paths:
                print(f"    {label_of(p, roots)}")
        if len(exact_dups) > 10:
            print(f"  ... 還有 {len(exact_dups) - 10} 組（看 {args.out}）")
        print(f"\n  其中跨類別/跨 split 衝突：{sum(1 for h,ps in exact_dups.items() if '跨' in classify(ps))} 組 ⚠️")
        for h, paths in exact_dups.items():
            report["exact_duplicates"].append({
                "hash": h,
                "kind": classify(paths),
                "paths": [str(p) for p in paths],
            })

    if near_dups:
        print(f"\n[視覺幾乎相同 - {len(near_dups)} 組]")
        for h, paths in list(near_dups.items())[:10]:
            kind = classify(paths)
            print(f"  {kind}")
            for p in paths:
                print(f"    {label_of(p, roots)}")
        if len(near_dups) > 10:
            print(f"  ... 還有 {len(near_dups) - 10} 組（看 {args.out}）")
        print(f"\n  其中跨類別/跨 split 衝突：{sum(1 for h,ps in near_dups.items() if '跨' in classify(ps))} 組 ⚠️")
        for h, paths in near_dups.items():
            report["near_duplicates"].append({
                "hash": h,
                "kind": classify(paths),
                "paths": [str(p) for p in paths],
            })

    if not exact_dups and not near_dups:
        print("\n沒找到重複，乾淨！")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n完整報告：{args.out}")


if __name__ == "__main__":
    main()
