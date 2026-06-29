"""
從 Roboflow Universe / Kaggle 下載公開傷口分割資料集，
解壓並按類別放到 data/raw/sam_collected/<Class>/ 供 AnyLabeling 標註。

支援來源：
    1. Roboflow Universe（需要 ROBOFLOW_API_KEY 環境變數）
    2. Kaggle dataset（需要 ~/.kaggle/kaggle.json）
    3. 直接從 URL 下載 zip

使用方式（範例）：
    # Roboflow
    python scripts/download_public_wound.py roboflow \
        --workspace wound-care --project cut-wound-detection --version 3 \
        --class-name Cut

    # Kaggle
    python scripts/download_public_wound.py kaggle \
        --dataset username/wound-segmentation-dataset \
        --class-name Abrasion

    # 直接 URL（任何 zip）
    python scripts/download_public_wound.py url \
        --url https://example.com/dataset.zip \
        --class-name Laceration
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST_ROOT = PROJECT_ROOT / "data" / "raw" / "sam_collected"
VALID_CLASSES = {
    "cut", "abrasion", "laceration", "footulcer", "chronic",
    "Cut", "Abrasion", "Laceration", "DFU", "Chronic",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def ensure_class(name: str) -> None:
    if name not in VALID_CLASSES:
        sys.exit(f"[err] class-name 必須是 {VALID_CLASSES} 之一，got: {name}")


def flatten_images(extract_dir: Path, dst_dir: Path) -> int:
    """把解壓後資料夾裡所有影像（不論深度）抓到 dst_dir，回傳張數。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in extract_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in IMG_EXTS:
            continue
        # 跳過明顯是 mask 的檔（如 _mask.png）
        if "mask" in p.stem.lower() or "label" in p.stem.lower():
            continue
        target = dst_dir / f"{p.parent.name}_{p.name}"
        # 避免同名衝突
        i = 1
        while target.exists():
            target = dst_dir / f"{p.parent.name}_{p.stem}_{i}{p.suffix}"
            i += 1
        shutil.copy2(p, target)
        count += 1
    return count


def cmd_roboflow(args: argparse.Namespace) -> None:
    """從 Roboflow Universe 下載。需要 pip install roboflow，並設好 API key。"""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("[err] 請先 `pip install roboflow`")

    api_key = os.environ.get("ROBOFLOW_API_KEY") or args.api_key
    if not api_key:
        sys.exit("[err] 請設 ROBOFLOW_API_KEY 環境變數或用 --api-key 傳入")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    # 下載 COCO 格式（只是用來解壓拿圖；mask 我們會用 SAM 重標）
    dataset = version.download("coco", location=str(args.tmp_dir))
    print(f"[ok] downloaded to {dataset.location}")

    dst = DEFAULT_DST_ROOT / args.class_name
    n = flatten_images(Path(dataset.location), dst)
    print(f"[ok] {n} images → {dst}")


def cmd_kaggle(args: argparse.Namespace) -> None:
    """從 Kaggle 下載 dataset，需要 kaggle CLI + ~/.kaggle/kaggle.json"""
    if shutil.which("kaggle") is None:
        sys.exit("[err] 請先 `pip install kaggle` 並設好 ~/.kaggle/kaggle.json")

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"[..] kaggle datasets download -d {args.dataset}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(tmp), "--unzip"],
        check=True,
    )

    dst = DEFAULT_DST_ROOT / args.class_name
    n = flatten_images(tmp, dst)
    print(f"[ok] {n} images → {dst}")


def cmd_url(args: argparse.Namespace) -> None:
    """從任意 URL 下載 zip 並解壓"""
    import urllib.request

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "download.zip"

    print(f"[..] downloading {args.url}")
    urllib.request.urlretrieve(args.url, zip_path)

    extract_dir = tmp / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    dst = DEFAULT_DST_ROOT / args.class_name
    n = flatten_images(extract_dir, dst)
    print(f"[ok] {n} images → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tmp-dir", default=str(PROJECT_ROOT / "data" / "raw" / "_tmp_download"),
                        help="解壓暫存目錄")

    sub = parser.add_subparsers(dest="source", required=True)

    # roboflow
    p_rf = sub.add_parser("roboflow", help="從 Roboflow Universe 下載")
    p_rf.add_argument("--workspace", required=True)
    p_rf.add_argument("--project", required=True)
    p_rf.add_argument("--version", type=int, required=True)
    p_rf.add_argument("--class-name", required=True, help=f"歸到哪一類：{VALID_CLASSES}")
    p_rf.add_argument("--api-key", default=None, help="若未設環境變數，由此傳入")
    p_rf.set_defaults(func=cmd_roboflow)

    # kaggle
    p_kg = sub.add_parser("kaggle", help="從 Kaggle 下載")
    p_kg.add_argument("--dataset", required=True, help="例如 username/wound-segmentation")
    p_kg.add_argument("--class-name", required=True)
    p_kg.set_defaults(func=cmd_kaggle)

    # url
    p_url = sub.add_parser("url", help="直接從 URL 抓 zip")
    p_url.add_argument("--url", required=True)
    p_url.add_argument("--class-name", required=True)
    p_url.set_defaults(func=cmd_url)

    args = parser.parse_args()
    ensure_class(args.class_name)
    DEFAULT_DST_ROOT.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
