"""测速：给定骨干和分辨率，测单卡训练吞吐，反推每轮耗时。

用法
----
python bench.py --backbone densenet121 --img-size 320 --batch-size 16
python bench.py --backbone densenet121 --img-size 320 --n-images 35000

只测 GPU（默认）＝ 纯算力上限，对应"图片已经预先缩放成小图"的情况；
加 --with-loader 会连 PNG 解码一起测，对应"直接读原始大图"的情况。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import NUM_CLASSES, build_model, eval_transform, load_rgb


def bench_gpu(backbone: str, img_size: int, batch_size: int, iters: int = 12) -> float:
    device = torch.device("cuda")
    model = build_model(backbone, NUM_CLASSES, pretrained=False).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.rand(batch_size, NUM_CLASSES, device=device)

    for _ in range(3):  # 预热
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch_size * iters / dt


def bench_loader(images: list[str], img_size: int, n: int = 30) -> float:
    """纯 CPU：读文件 + 解码 + resize，不算 GPU。"""
    tf = eval_transform(img_size)
    t0 = time.perf_counter()
    for p in images[:n]:
        _ = tf(load_rgb(p))
    dt = time.perf_counter() - t0
    return n / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="densenet121")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=30000, help="训练集图片总数，用于反推每轮耗时")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--with-loader", action="store_true", help="同时测 PNG 解码吞吐")
    ap.add_argument("--sample-dir", default=None, help="--with-loader 时用的图片目录")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  {props.total_memory/1024**3:.1f} GB  "
          f"sm{props.major}{props.minor}  {props.multi_processor_count} SMs")
    print(f"配置: {args.backbone} @{args.img_size}px  batch={args.batch_size}  AMP=fp16")

    ips = bench_gpu(args.backbone, args.img_size, args.batch_size)
    epoch_s = args.n_images / ips
    print(f"\nGPU 训练吞吐: {ips:.1f} img/s")
    print(f"每轮（{args.n_images} 张）: {epoch_s/60:.1f} 分钟")
    print(f"{args.epochs} 轮: {epoch_s*args.epochs/3600:.1f} 小时")

    if args.with_loader:
        if not args.sample_dir:
            raise SystemExit("--with-loader 需要 --sample-dir")
        imgs = [str(p) for p in Path(args.sample_dir).rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}][:200]
        if not imgs:
            raise SystemExit(f"{args.sample_dir} 下没找到图片")
        cpu_ips = bench_loader(imgs, args.img_size)
        # 4 个 worker 并行解码，理想情况线性放大
        for workers in (4, 8):
            eff = cpu_ips * workers
            print(f"\n单核解码 {cpu_ips:.1f} img/s -> {workers} worker 理想 {eff:.0f} img/s")
            bottleneck = "GPU" if ips < eff else "数据管道（解码/IO）"
            real = min(ips, eff * 0.8)
            print(f"  实际瓶颈: {bottleneck}；估计有效吞吐 {real:.0f} img/s -> "
                  f"每轮 {args.n_images/real/60:.1f} 分钟")


if __name__ == "__main__":
    main()
