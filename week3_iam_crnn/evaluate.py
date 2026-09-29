# -*- coding: utf-8 -*-
"""第3周评估脚本:测试集 CER/WER + 正确与失败案例可视化。

任务书要求「对识别结果进行可视化展示」并「展示若干正确案例和失败案例分析」,
本脚本即为该验收项的产出。

用法(在 week3_iam_crnn 目录下,需先跑完 train.py):
    python evaluate.py                                  # 用最新的模型
    python evaluate.py --tag crnn_h256 --samples 8      # 每个案例展示 8 个
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import get_device, get_iam_loaders, BLANK, PROJECT_ROOT as _PR
from models import build_crnn, greedy_decode
from train import compute_cer_wer


def main():
    ap = argparse.ArgumentParser(description="第3周 CRNN 评估与案例可视化")
    ap.add_argument("--tag", type=str, default="", help="outputs/week3 下的模型目录名;留空则自动选最新")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=8, help="正确/失败案例各展示多少个")
    ap.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    args = ap.parse_args()

    # ---- 找到要评估的模型目录 ----
    if args.tag:
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "week3", args.tag)
    else:
        cands = [d for d in glob.glob(os.path.join(PROJECT_ROOT, "outputs", "week3", "*"))
                 if os.path.exists(os.path.join(d, "best_model.pt"))]
        if not cands:
            print("未找到已训练的模型,请先运行 train.py")
            return
        out_dir = max(cands, key=os.path.getmtime)
    tag = os.path.basename(out_dir)
    print(f"评估模型: {tag}")

    device = get_device()
    _, _, test_loader, vocab = get_iam_loaders(args.data_dir, batch_size=args.batch_size)

    model = build_crnn(len(vocab)).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.eval()

    # ---- 跑测试集,收集预测 ----
    preds, trues = [], []
    with torch.no_grad():
        for x, targets, target_lengths, texts in test_loader:
            logits = model(x.to(device))
            decoded = greedy_decode(logits.log_softmax(2), blank=BLANK)
            preds.extend(vocab.decode(d) for d in decoded)
            trues.extend(texts)

    cer, wer = compute_cer_wer(preds, trues)
    print(f"\n测试集样本数 = {len(trues)}")
    print(f"CER(字符错误率) = {cer * 100:.2f}%")
    print(f"WER(单词错误率) = {wer * 100:.2f}%")

    # ---- 分类:完全正确 / 错误 ----
    correct = [(p, t) for p, t in zip(preds, trues) if p == t]
    wrong = [(p, t) for p, t in zip(preds, trues) if p != t]
    print(f"完全正确 = {len(correct)} ({len(correct) / len(trues) * 100:.2f}%)")
    print(f"识别错误 = {len(wrong)} ({len(wrong) / len(trues) * 100:.2f}%)")

    # ---- 可视化:从测试集里按文本取对应图像 ----
    # 建立 文本 -> 图像路径 的索引(batch 里没带路径,这里回查一次)
    iam_dir = os.path.join(PROJECT_ROOT, "data", "IAM")
    from common import parse_words_txt, form_id_of, image_path_of, load_splits
    splits = load_splits(os.path.join(iam_dir, "splits"))
    path_by_text = {}
    for wid, txt in parse_words_txt(os.path.join(iam_dir, "words.txt")):
        if form_id_of(wid) in splits["test"]:
            p = image_path_of(iam_dir, wid)
            if os.path.exists(p):
                path_by_text.setdefault(txt, p)

    def draw_cases(cases, title, fname, color):
        n = min(args.samples, len(cases))
        if n == 0:
            return
        fig, axes = plt.subplots(n, 1, figsize=(9, 1.15 * n))
        if n == 1:
            axes = [axes]
        for ax, (p, t) in zip(axes, cases[:n]):
            img_path = path_by_text.get(t)
            if img_path:
                ax.imshow(Image.open(img_path).convert("L"), cmap="gray", aspect="auto")
            else:
                ax.text(0.5, 0.5, "(image not found)", ha="center")
            ax.set_title(f"true: {t!r}    pred: {p!r}", fontsize=9, color=color)
            ax.axis("off")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"已保存: {out}")

    draw_cases(correct, f"Correctly Recognized ({len(correct)} total, showing {min(args.samples, len(correct))})",
               "correct_cases.png", "green")
    draw_cases(wrong, f"Failed Cases ({len(wrong)} total, showing {min(args.samples, len(wrong))})",
               "failed_cases.png", "red")

    # ---- 保存指标 ----
    metrics = {"tag": tag, "test_cer": cer, "test_wer": wer,
               "n_test": len(trues), "n_correct": len(correct), "n_wrong": len(wrong)}
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {os.path.join(out_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
