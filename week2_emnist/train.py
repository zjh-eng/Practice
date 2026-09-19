"""第2周训练脚本:在 EMNIST 上训练 MLP / CNN / ResNet18,并记录对比指标。

为保证三种模型可公平对比,除网络结构外其余超参数完全一致;
每个模型训练结束后会把「准确率 / 训练时间 / 参数量」写入 result.json,
供 compare.py 汇总成对比表。

用法(在 week2_emnist 目录下):
    python train.py --model mlp
    python train.py --model cnn --augment
    python train.py --model resnet18 --epochs 15
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import get_device, set_seed, get_emnist_loaders, DEFAULT_SPLIT
from models import build_model, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """跑一个 epoch。optimizer 为 None 时是评估模式(不更新参数)。"""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, ncols=90)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(acc=f"{correct / total:.3f}")
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser(description="第2周 EMNIST 字符识别训练")
    ap.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "resnet18"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true", help="对训练集启用数据增强")
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    args = ap.parse_args()

    tag = args.model + ("_aug" if args.augment else "")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "week2", tag)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"模型={args.model}  增强={args.augment}  设备={device}")

    train_loader, val_loader, test_loader, class_names = get_emnist_loaders(
        args.data_dir, split=args.split, batch_size=args.batch_size,
        augment=args.augment, seed=args.seed)
    num_classes = len(class_names)
    print(f"类别数={num_classes}  训练批次数={len(train_loader)}")

    model = build_model(args.model, num_classes=num_classes).to(device)
    n_params = count_parameters(model)
    print(f"可训练参数量={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt = os.path.join(out_dir, "best_model.pt")

    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, desc=f"train {epoch}/{args.epochs}")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                    desc=f"val   {epoch}/{args.epochs}")
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  epoch {epoch:2d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt)
    train_time = time.perf_counter() - t_start

    # ---- 用最优权重在测试集上评估 ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc = run_epoch(model, test_loader, criterion, device, desc="test")
    print(f"\n最优验证准确率 = {best_val_acc:.4f}")
    print(f"测试集准确率   = {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"训练总耗时     = {train_time:.1f}s ({train_time / args.epochs:.1f}s/epoch)")
    print(f"参数量         = {n_params:,}")

    # ---- 画训练曲线(图内用英文,避免中文字体缺失) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{tag} - Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{tag} - Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)

    # ---- 保存结果(供 compare.py 汇总) ----
    result = {
        "model": args.model,
        "tag": tag,
        "augment": bool(args.augment),
        "split": args.split,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_time_sec": train_time,
        "sec_per_epoch": train_time / args.epochs,
        "history": history,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {os.path.join(out_dir, 'result.json')}")


if __name__ == "__main__":
    main()
