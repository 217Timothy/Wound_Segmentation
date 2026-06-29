"""
產生視覺化重複預覽 HTML，讓你用人眼確認哪些圖該刪。

掃 dhash 視覺幾乎相同 + 跨類別衝突，把每組相似圖縮圖並排塞進 HTML。
你在瀏覽器打開後，勾選要刪的圖，按按鈕匯出刪除清單（.sh 腳本）。

用法：
    python tools/review_duplicates.py \
        --root data/raw/multiclass/abrasion \
        --root data/raw/multiclass/cut \
        --root data/raw/multiclass/laceration \
        --out outputs/dup_review.html

    # 然後在瀏覽器打開 outputs/dup_review.html
    # 看完勾選後按「下載刪除腳本」→ bash delete_duplicates.sh

旗標：
    --threshold N    dhash 漢明距離閾值（預設 8，越小越嚴格）
    --thumb-size N   縮圖大小（預設 200px）
"""
from __future__ import annotations

import argparse
import base64
import html
import io
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            px = list(im.getdata())
        h = 0
        for r in range(size):
            for c in range(size):
                i = r * (size + 1) + c
                h = (h << 1) | (1 if px[i] > px[i + 1] else 0)
        return h
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def make_thumb_b64(path: Path, max_size: int = 200) -> str:
    """讀圖 → 等比縮圖 → base64 jpeg，方便嵌進 HTML"""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def cluster_similar(files: list[tuple[Path, int, str]], threshold: int) -> list[list[int]]:
    """傳回 group 列表，每個 group 是 indices。只回 size>1 的 group。"""
    n = len(files)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(files[i][1], files[j][1]) <= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


HTML_HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="utf-8">
<title>重複資料審查</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background:#fafafa; color:#222; margin:0; padding:24px; }
  h1 { margin-top: 0; }
  .summary { background:#fff; padding:16px 20px; border-radius:8px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom:24px; }
  .controls { position: sticky; top:0; background:#fff; padding:12px 16px;
              border:1px solid #ddd; border-radius:8px; margin-bottom:16px;
              display:flex; gap:12px; align-items:center; z-index:10;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  button { padding:8px 16px; font-size:14px; border-radius:6px; cursor:pointer;
           border:1px solid #888; background:#fff; }
  button.primary { background:#2563eb; color:#fff; border-color:#2563eb; }
  .group { background:#fff; border:1px solid #e0e0e0; border-radius:8px;
           margin-bottom:16px; padding:14px; }
  .group h3 { margin:0 0 10px 0; font-size:15px; }
  .group.warn { border-color:#dc2626; background:#fef2f2; }
  .group.warn h3 { color:#b91c1c; }
  .imgs { display:flex; flex-wrap:wrap; gap:12px; }
  .card { width:220px; text-align:center; padding:8px; border:1px solid #eee;
          border-radius:6px; background:#fafafa; }
  .card img { width:100%; height:auto; display:block; border-radius:4px; }
  .card label { display:flex; align-items:center; gap:6px; font-size:12px;
                margin-top:6px; cursor:pointer; }
  .card .path { font-family: ui-monospace, monospace; font-size:11px;
                color:#555; word-break:break-all; margin-top:4px; }
  .delete .card img { opacity:.4; }
  details { margin-top:8px; }
  code { background:#eef; padding:2px 6px; border-radius:4px; font-size:12px; }
  #count { font-weight:bold; color:#dc2626; }
</style></head><body data-project-root="__PROJECT_ROOT__">
<h1>重複資料審查</h1>
"""

HTML_SCRIPT = """
<script>
function updateCount() {
  const n = document.querySelectorAll('input[type="checkbox"]:checked').length;
  document.getElementById('count').textContent = n;
  document.querySelectorAll('.card').forEach(card => {
    const cb = card.querySelector('input[type="checkbox"]');
    if (cb && cb.checked) card.classList.add('delete');
    else card.classList.remove('delete');
  });
}
document.addEventListener('change', e => {
  if (e.target.matches('input[type="checkbox"]')) updateCount();
});

function autoSelectExceptFirst() {
  document.querySelectorAll('.group').forEach(g => {
    const cbs = g.querySelectorAll('input[type="checkbox"]');
    cbs.forEach((cb, i) => cb.checked = (i > 0));
  });
  updateCount();
}
function clearAll() {
  document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
  updateCount();
}
function downloadScript() {
  const paths = [];
  document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
    paths.push(cb.dataset.path);
  });
  if (paths.length === 0) { alert('沒勾任何要刪的圖'); return; }
  const projectRoot = document.body.dataset.projectRoot || '$HOME/Desktop/ITRI_Segmentation_Project';
  const lines = ['#!/usr/bin/env bash',
                 '# 由 review_duplicates.py 產生',
                 '# 自動產生時間: ' + new Date().toISOString(),
                 'set -e',
                 'PROJECT_ROOT="' + projectRoot + '"',
                 'cd "$PROJECT_ROOT"',
                 '',
                 'echo "將從 $PROJECT_ROOT 刪除 ' + paths.length + ' 個檔案"',
                 'read -p "確定？(y/N) " ans',
                 '[[ $ans == [yY] ]] || { echo "已取消"; exit 1; }',
                 ''];
  paths.forEach(p => lines.push('rm -v ' + JSON.stringify(p)));
  const blob = new Blob([lines.join('\\n') + '\\n'], {type:'text/x-shellscript'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'delete_duplicates.sh';
  a.click();
}
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", required=True,
                        help="要掃的資料夾（可重複指定，每個資料夾名稱會被當作類別名）")
    parser.add_argument("--out", default="outputs/dup_review.html")
    parser.add_argument("--threshold", type=int, default=8,
                        help="dhash 漢明距離閾值，越小越嚴格（預設 8）")
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--project-root", default=None,
                        help="生成的刪除腳本將 cd 到此路徑（預設用此檔的 parent.parent）。"
                             "讓刪除腳本路徑為相對路徑，跨機器可執行。")
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 推算 project root：給 HTML 的 JS 用，產出相對路徑的刪除腳本
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]
    print(f"[info] project_root = {project_root}")

    # 收圖 + 算 hash
    print("[..] 算 dhash...")
    all_files: list[tuple[Path, int, str]] = []   # (path, hash, class_label)
    per_class: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    for r in roots:
        cls = r.name
        n = 0
        for p in sorted(r.iterdir()):
            if p.suffix not in IMG_EXTS:
                continue
            h = dhash(p)
            if h is None:
                continue
            entry = (p, h, cls)
            all_files.append(entry)
            per_class[cls].append(entry)
            n += 1
        print(f"     {cls}: {n} 張")

    # 同類內 group
    print("[..] 分群（同類內）...")
    intra_groups: list[tuple[str, list[tuple[Path, int, str]]]] = []
    for cls, files in per_class.items():
        for g_indices in cluster_similar(files, args.threshold):
            members = [files[i] for i in g_indices]
            intra_groups.append((cls, members))

    # 跨類別 group（找出至少含兩個類別的相似團）
    print("[..] 分群（跨類別衝突）...")
    cross_groups: list[list[tuple[Path, int, str]]] = []
    for g_indices in cluster_similar(all_files, args.threshold):
        members = [all_files[i] for i in g_indices]
        classes = {m[2] for m in members}
        if len(classes) > 1:
            cross_groups.append(members)

    # 產 HTML
    print("[..] 產生 HTML...")
    # 推測使用者實際的 project root（Mac 桌面路徑），讓 JS 產出的腳本路徑能在使用者機器上跑
    user_project_root = "$HOME/Desktop/" + project_root.name
    head = HTML_HEAD_TEMPLATE.replace("__PROJECT_ROOT__", html.escape(user_project_root))
    parts = [head]

    # 摘要
    parts.append(f"""
    <div class="summary">
      <p>掃描資料夾：{', '.join(f'<code>{r.name}</code>' for r in roots)}</p>
      <p>dhash 閾值：≤ {args.threshold} bits（差異越小越像）</p>
      <p>同類別內相似組：<b>{len(intra_groups)}</b>　|
         跨類別衝突組：<b style="color:#dc2626">{len(cross_groups)}</b></p>
      <p style="color:#555;font-size:13px">
        勾選你想<b>刪除</b>的圖，按下方按鈕匯出 <code>.sh</code> 腳本，再去終端機跑。
      </p>
    </div>
    <div class="controls">
      <button onclick="autoSelectExceptFirst()">每組保留第一張，其餘自動勾選</button>
      <button onclick="clearAll()">全部清除勾選</button>
      <button class="primary" onclick="downloadScript()">下載刪除腳本</button>
      <span style="margin-left:auto">已勾選刪除：<span id="count">0</span> 張</span>
    </div>
    """)

    def render_card(p: Path, cls: str) -> str:
        b64 = make_thumb_b64(p, args.thumb_size)
        # 用相對於 project_root 的路徑，讓刪除腳本可以跨機器用
        try:
            rel_path = str(p.relative_to(project_root))
        except ValueError:
            rel_path = str(p)  # 不在 project_root 下就用絕對路徑
        return f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{b64}" alt="{html.escape(p.name)}">
          <div class="path">{html.escape(cls)}/{html.escape(p.name)}</div>
          <label><input type="checkbox" data-path="{html.escape(rel_path)}"> 刪除</label>
        </div>"""

    # 跨類別衝突放最前面（最嚴重）
    if cross_groups:
        parts.append("<h2>⚠️ 跨類別衝突（同張圖被分到不同類別）</h2>")
        for i, members in enumerate(cross_groups, 1):
            classes = sorted({m[2] for m in members})
            parts.append(f'<div class="group warn"><h3>衝突 #{i}：{" ↔ ".join(classes)}（{len(members)} 張）</h3>')
            parts.append('<div class="imgs">')
            for (p, _, cls) in members:
                parts.append(render_card(p, cls))
            parts.append('</div></div>')

    # 同類內相似
    if intra_groups:
        parts.append("<h2>同類別內視覺相似</h2>")
        # 依類別分組顯示
        by_class: dict[str, list[list[tuple[Path, int, str]]]] = defaultdict(list)
        for cls, members in intra_groups:
            by_class[cls].append(members)
        for cls in sorted(by_class):
            groups = by_class[cls]
            parts.append(f"<h3 style='margin-top:24px'>📁 {cls}（{len(groups)} 組相似）</h3>")
            for i, members in enumerate(groups, 1):
                parts.append(f'<div class="group"><h3>{cls} 相似組 #{i}（{len(members)} 張）</h3>')
                parts.append('<div class="imgs">')
                for (p, _, c) in members:
                    parts.append(render_card(p, c))
                parts.append('</div></div>')

    if not intra_groups and not cross_groups:
        parts.append("<div class='summary'><p>沒找到相似圖，乾淨！</p></div>")

    parts.append(HTML_SCRIPT)
    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"[ok] 寫入 {out_path}")
    print(f"     檔案大小：{out_path.stat().st_size / 1024:.1f} KB")
    print(f"     在 Finder/檔案總管打開，或用瀏覽器直接開")


if __name__ == "__main__":
    main()
