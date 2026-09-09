"""第1周评估脚本:测试集准确率 + 混淆矩阵 + 错样本分析。

用法(在 week1_mnist 目录下执行,需先跑完 train.py 得到模型):
    python evaluate.py
"""
import argparse
import os
from collections import Counter

# OMP 冲突规避(需在 import torch 之前)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 项目根目录(由脚本位置推导,不受 PyCharm「工作目录」设置影响)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from common import get_device, get_mnist_loaders
from model import MNIST_CNN


def main():
    parser = argparse.ArgumentParser(description="第1周 MNIST 评估")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "week1"))
    parser.add_argument("--ckpt", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "week1", "best_model.pt"))
    parser.add_argument("--num-errors", type=int, default=12, help="可视化错样本数量")
    args = parser.parse_args()

    device = get_device()
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    _, _, test_loader = get_mnist_loaders(args.data_dir, args.batch_size)

    all_preds, all_labels = [], []
    errors = []  # (image, true_label, pred_label)

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = logits.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
            for img, true, pred in zip(x.cpu(), y, preds):
                if true != pred:
                    errors.append((img, true.item(), pred.item()))

    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    print(f"测试集准确率: {acc:.4f} ({acc * 100:.2f}%)")

    # ============ 错误样本统计分析 ============
    # 1) 每个数字被错判的次数(反映哪些数字最难识别)
    err_by_true = Counter(t for _, t, _ in errors)
    print("\n=== 错误样本分析 ===")
    print("各数字被错判次数(真值 -> 错判数):")
    for d in range(10):
        print(f"  数字 {d}: {err_by_true.get(d, 0)} 次")

    # 2) 最常见的混淆对(真值 -> 预测),按次数降序
    confusion_pairs = Counter((t, p) for _, t, p in errors)
    print(f"\n最常见的混淆对(共 {len(confusion_pairs)} 种,列出前 10):")
    for (t, p), cnt in confusion_pairs.most_common(10):
        print(f"  {t} -> {p}: {cnt} 次")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("MNIST Confusion Matrix")
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    print(f"混淆矩阵已保存: {os.path.join(args.out_dir, 'confusion_matrix.png')}")

    # 错样本可视化(固定 3x4 网格)
    n = min(args.num_errors, len(errors))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for i, (img, true, pred) in enumerate(errors[:n]):
        ax = axes[i // 4][i % 4]
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.set_title(f"True:{true} Pred:{pred}", color="red")
        ax.axis("off")
    fig.suptitle(f"Misclassified Examples ({len(errors)} errors, showing first {n})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "error_analysis.png"), dpi=150)
    print(f"错样本可视化已保存: {os.path.join(args.out_dir, 'error_analysis.png')}")


if __name__ == "__main__":
    main()
