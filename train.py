"""训练：多标签 BCE + AMP + 余弦退火 + EMA，按验证集宏平均 AUC 选最优。

用法（数据体检产出 manifest 之后）
----------------------------------
python train.py --train-manifest runs/data_check/train/manifest.csv \
                --val-manifest   runs/data_check/val/manifest.csv \
                --backbone densenet121.tv_in1k --img-size 320 \
                --epochs 8 --batch-size 16 --out-dir runs/train

只用一份 manifest、没有独立验证集时：
python train.py --train-manifest all.csv --val-frac 0.15   # 按 Subject_id 分组切，避免同一患者跨集
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    ChestDataset,
    build_model,
    eval_transform,
    safe_macro_auc,
    seed_everything,
    train_transform,
)


class ModelEMA:
    """指数滑动平均，几乎免费地稳定验证指标。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.ema[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return self.ema


def group_split(manifest: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = "Subject_id" if "Subject_id" in manifest.columns and manifest["Subject_id"].notna().any() else "Study_id"
    ids = manifest[key].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(ids[:n_val])
    mask = manifest[key].astype(str).isin(val_ids)
    return manifest[~mask].reset_index(drop=True), manifest[mask].reset_index(drop=True)


def make_loader(manifest, transform, batch_size, workers, shuffle, num_classes):
    ds = ChestDataset(manifest, transform, num_classes)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )


class Prefetcher:
    """用一个后台线程提前取 batch，让读图/解码和 GPU 计算重叠。

    Windows 上 DataLoader 的 num_workers>0 依赖进程间命名管道，在受限沙箱里会被拒；
    实测单线程读图 137 img/s，叠上 GPU 却只有 ~67 img/s。用线程预取可以把这个损失收回来
    （PIL 解码会释放 GIL，所以线程确实能并行）。
    """

    def __init__(self, loader, ahead: int = 4):
        self.loader = loader
        self.ahead = ahead

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.ahead)
        sentinel = object()

        def produce():
            try:
                for batch in self.loader:
                    q.put(batch)
            except Exception as exc:  # 把异常带回主线程
                q.put(exc)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=produce, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        t.join(timeout=1)

    def __len__(self):
        return len(self.loader)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, np.ndarray, np.ndarray, float]:
    model.eval()
    probs, targets, losses = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(float(loss))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.cpu().numpy())
    p, t = np.concatenate(probs), np.concatenate(targets)
    auc, _ = safe_macro_auc(t, p)
    return auc, p, t, float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--backbone", default="densenet121", help="torchvision 短名或 timm 全名")
    ap.add_argument("--weights-path", default=None, help="本地预训练权重（离线时用）")
    ap.add_argument("--class-ids", default=None, help="手工指定官方类别 id，如 1,2,...,10")
    ap.add_argument("--init-ckpt", default=None, help="从这个 ckpt 的权重继续训练（用于换分辨率精调）")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 子进程数；受限沙箱里用 0")
    ap.add_argument("--prefetch", type=int, default=4, help="后台线程预取批次数，workers=0 时补性能")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", default="runs/train")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} backbone={args.backbone} img_size={args.img_size}")

    train_df = pd.read_csv(args.train_manifest)
    if args.val_manifest:
        val_df = pd.read_csv(args.val_manifest)
        print(f"train 行 {len(train_df)} / val 行 {len(val_df)}（官方划分）")
    else:
        train_df, val_df = group_split(train_df, args.val_frac, args.seed)
        print(f"无独立验证集，按患者分组切出 {len(val_df)} 行作验证")

    # 官方类别 id（本赛事是 1..10，不是 0..9）：从 data_check 产出的 class_ids.json 读
    cid_path = Path(args.train_manifest).parent / "class_ids.json"
    if cid_path.exists():
        class_ids = json.loads(cid_path.read_text(encoding="utf-8"))
    elif args.class_ids:
        class_ids = [int(x) for x in args.class_ids.split(",")]
    else:
        class_ids = list(range(1, 11))
    num_classes = len(class_ids)
    print(f"官方类别 id: {class_ids}（内部下标 0..{num_classes-1}）")

    train_loader = make_loader(train_df, train_transform(args.img_size), args.batch_size, args.workers, True, num_classes)
    val_loader = make_loader(val_df, eval_transform(args.img_size), args.batch_size, args.workers, False, num_classes)

    model = build_model(args.backbone, num_classes, pretrained=not args.no_pretrained,
                        weights_path=args.weights_path).to(device)
    if args.init_ckpt:
        # 从已有权重继续（例如 320 训完再在 384 上精调）
        prev = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(prev["model"])
        print(f"从 {args.init_ckpt} 初始化权重（该 ckpt val_auc={prev.get('val_macro_auc')}）")

    # 类别不均衡：用训练集阳性率算 pos_weight（上限 20，避免早期梯度爆炸）
    ycols = [f"y{c}" for c in range(num_classes)]
    per_study = train_df.drop_duplicates("Study_id")
    pos = per_study[ycols].sum().to_numpy(dtype=np.float64)
    total = max(len(per_study), 1)
    pos_weight = np.clip((total - pos) / np.maximum(pos, 1.0), 1.0, 20.0)
    print("pos_weight:", np.round(pos_weight, 2).tolist())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))

    decay_params = [p for n, p in model.named_parameters() if p.ndim > 1 and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.ndim <= 1 and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": other_params, "weight_decay": 0.0}],
        lr=args.lr,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay)

    log_path = out / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_macro_auc", "val_auc_ema", "lr", "seconds"])

    best_auc, best_epoch, global_step = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y, _) in enumerate(Prefetcher(train_loader, args.prefetch) if args.prefetch > 0 else train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            # EMA 热身：训练早期用较小的 decay，否则前几百步的 EMA 严重滞后于真实权重
            ema.decay = min(args.ema_decay, (1 + global_step) / (10 + global_step))
            ema.update(model)
            running += float(loss)
            if (i + 1) % max(1, steps_per_epoch // 5) == 0:
                print(f"  epoch {epoch} step {i+1}/{steps_per_epoch} loss={float(loss):.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        train_loss = running / steps_per_epoch
        val_auc, _, _, val_loss = evaluate(model, val_loader, device, criterion)

        # EMA 权重也评估一次，取更好的那个存
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        ema_auc, ema_probs, ema_targets, ema_loss = evaluate(model, val_loader, device, criterion)
        model.load_state_dict(raw_state)

        use_ema = ema_auc >= val_auc
        cur_auc = ema_auc if use_ema else val_auc
        secs = time.time() - t0
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f} ema_auc={ema_auc:.4f} -> 采用{'EMA' if use_ema else 'raw'} {secs:.0f}s",
              flush=True)

        with log_path.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                    f"{val_auc:.6f}", f"{ema_auc:.6f}",
                                    f"{scheduler.get_last_lr()[0]:.3e}", f"{secs:.1f}"])

        if cur_auc > best_auc:
            best_auc, best_epoch = cur_auc, epoch
            ckpt = {
                "model": (ema.state_dict() if use_ema else raw_state),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "num_classes": num_classes,
                "class_ids": class_ids,
                "val_macro_auc": float(cur_auc),
                "epoch": epoch,
                "ema": use_ema,
            }
            torch.save(ckpt, out / "best.pt")
            print(f"  -> 保存 best.pt (val_macro_auc={cur_auc:.4f})")

    summary = {
        "best_val_macro_auc": float(best_auc),
        "best_epoch": best_epoch,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "pos_weight": [round(float(v), 3) for v in pos_weight],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成。最佳 val Macro-AUC = {best_auc:.4f} (epoch {best_epoch})")
    print(f"权重: {out/'best.pt'}\n日志: {log_path}")


if __name__ == "__main__":
    main()
