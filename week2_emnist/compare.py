# -*- coding: utf-8 -*-
"""汇总各模型结果,生成第2周验收物:实验对比表 + 对比图。

读取 outputs/week2/*/result.json(由 train.py 产生),
输出:
    outputs/week2/comparison.md    对比表(Markdown)
    outputs/week2/comparison.png   对比柱状图

用法(在 week2_emnist 目录下,需先跑完若干 train.py):
    python compare.py
"""
import glob
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "week2")
MODEL_ORDER = {"mlp": 0, "cnn": 1, "resnet18": 2}
MODEL_LABEL = {"mlp": "MLP", "cnn": "CNN", "resnet18": "ResNet18"}


def load_results():
    results = []
    for path in glob.glob(os.path.join(OUT_DIR, "*", "result.json")):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    results.sort(key=lambda r: (MODEL_ORDER.get(r["model"], 9), r["augment"]))
    return results


def fmt_params(n):
    return f"{n / 1e6:.3f} M" if n >= 1e6 else f"{n / 1e3:.1f} K"


def main():
    results = load_results()
    if not results:
        print("未找到任何 result.json,请先运行 train.py 训练模型。")
        return

    # ---------- 对比表 ----------
    header = "| 模型 | 数据增强 | 测试准确率 | 训练时间(s) | 每轮耗时(s) | 参数量 |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {MODEL_LABEL.get(r['model'], r['model'])} "
            f"| {'是' if r['augment'] else '否'} "
            f"| {r['test_acc'] * 100:.2f}% "
            f"| {r['train_time_sec']:.1f} "
            f"| {r['sec_per_epoch']:.1f} "
            f"| {fmt_params(r['params'])} |"
        )
    table_md = "\n".join(lines)

    md_path = os.path.join(OUT_DIR, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 第2周 EMNIST 实验对比表\n\n")
        f.write(f"数据集:EMNIST {results[0]['split']}({results[0]['num_classes']} 类)\n\n")
        f.write(f"训练轮数:{results[0]['epochs']}  batch size:{results[0]['batch_size']}  "
                f"学习率:{results[0]['lr']}\n\n")
        f.write(table_md + "\n")

    print("=" * 70)
    print(table_md)
    print("=" * 70)
    print(f"\n对比表已保存: {md_path}")

    # ---------- 对比图 ----------
    labels = [r["tag"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    times = [r["train_time_sec"] for r in results]
    params = [r["params"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = range(len(labels))

    axes[0].bar(x, accs, color="#4C72B0")
    axes[0].set_title("Test Accuracy (%)")
    axes[0].set_ylim(min(accs) - 5 if min(accs) > 5 else 0, 100)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(labels, rotation=20)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)

    axes[1].bar(x, times, color="#DD8452")
    axes[1].set_title("Training Time (s)")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels, rotation=20)
    for i, v in enumerate(times):
        axes[1].text(i, v + max(times) * 0.01, f"{v:.0f}", ha="center", fontsize=9)

    axes[2].bar(x, params, color="#55A868")
    axes[2].set_yscale("log")
    axes[2].set_title("Parameters (log scale)")
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(labels, rotation=20)
    for i, v in enumerate(params):
        axes[2].text(i, v * 1.15, fmt_params(v), ha="center", fontsize=9)

    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "comparison.png")
    fig.savefig(png_path, dpi=150)
    print(f"对比图已保存: {png_path}")


if __name__ == "__main__":
    main()
