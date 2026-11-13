"""把原始大图缓存成小尺寸 JPEG，训练时不再解码 6MP 原图。

实测（本赛事数据，单核）：6MP JPEG 普通解码 14.8 img/s，draft 降采样 31.4 img/s；
32 位机器上 8 核并行做一遍缓存约 3~5 分钟，之后每轮读盘从 ~50GB 降到 ~2GB。

用法
----
python make_cache.py --src-manifest runs\\data_check\\train\\manifest.csv --out-dir cache\\train --size 384 --workers 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_one(job: tuple[str, str, int, int]) -> tuple[bool, int]:
    src, dst, size, quality = job
    dst_path = Path(dst)
    if dst_path.exists():
        return True, dst_path.stat().st_size
    try:
        im = Image.open(src)
        # draft 让 JPEG 解码器直接降采样，比先解全尺寸再 resize 快 2 倍以上
        im.draft("RGB", (size, size))
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.BILINEAR)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, "JPEG", quality=quality, optimize=False)
        return True, dst_path.stat().st_size
    except Exception:
        return False, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--manifest-out", default=None, help="输出 manifest，默认 <out-dir>/../manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.src_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    new_paths = []
    for p in df["image_path"].astype(str):
        src = Path(p)
        # 保持层级：subject_xxx/study_yyy/pic.jpg -> cache/subject_xxx/study_yyy/pic.jpg
        parts = src.parts
        rel = Path(*parts[-3:]) if len(parts) >= 3 else Path(src.name)
        dst = out_dir / rel.with_suffix(".jpg")
        jobs.append((str(src), str(dst), args.size, args.quality))
        new_paths.append(str(dst))

    print(f"待转换 {len(jobs)} 张 -> {args.size}px JPEG q{args.quality}，{args.workers} 线程")
    t0 = time.time()
    ok = 0
    total_bytes = 0
    # 用线程而不是进程：PIL 解码/编码会释放 GIL，且 DSH 沙箱禁止子进程的命名管道
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (succ, size) in enumerate(ex.map(convert_one, jobs, chunksize=64), 1):
            if succ:
                ok += 1
                total_bytes += size
            if i % 5000 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(jobs)}  {rate:.0f} img/s  已用 {(time.time()-t0)/60:.1f} min", flush=True)

    dt = time.time() - t0
    print(f"完成 {ok}/{len(jobs)} 张，{dt/60:.1f} 分钟，输出 {total_bytes/1024**3:.2f} GB")

    df_out = df.copy()
    df_out["image_path"] = new_paths
    manifest_out = Path(args.manifest_out) if args.manifest_out else out_dir / "manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(manifest_out, index=False, encoding="utf-8-sig")
    print(f"新 manifest: {manifest_out}")

    # class_ids 顺手带过去
    cid = Path(args.src_manifest).parent / "class_ids.json"
    if cid.exists():
        (manifest_out.parent / "class_ids.json").write_text(cid.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"class_ids: {manifest_out.parent/'class_ids.json'}")


if __name__ == "__main__":
    main()
