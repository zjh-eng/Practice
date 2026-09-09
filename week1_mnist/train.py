"""第1周训练脚本:训练 MNIST_CNN,画 loss/acc 曲线,保存最优模型。

用法(在 week1_mnist 目录下执行):
    python train.py --epochs 5 --batch-size 64 --lr 1e-3
"""
import argparse
import os

# OMP 冲突规避(需在 import torch 之前,见 common.py 注释)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 项目根目录(由脚本位置推导,不受 PyCharm「工作目录」设置影响)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 不弹窗,直接保存图片(想交互显示可删掉这行)
import matplotlib.pyplot as plt

from common import get_device, set_seed, get_mnist_loaders
from model import MNIST_CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch,返回平均 loss 和准确率。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估一个 epoch(不计算梯度),返回平均 loss 和准确率。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="第1周 MNIST CNN 训练")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "week1"))
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"使用设备: {device}")

    train_loader, val_loader, _ = get_mnist_loaders(args.data_dir, args.batch_size)

    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

    # 绘制训练 / 验证曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150)

    print(f"训练曲线已保存: {os.path.join(args.out_dir, 'training_curves.png')}")
    print(f"最优验证准确率: {best_val_acc:.4f}")
    print(f"模型已保存: {os.path.join(args.out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
