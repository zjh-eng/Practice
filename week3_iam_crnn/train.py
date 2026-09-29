"""第3周训练脚本:CRNN + CTC 单词级手写文字识别。

用法(在 week3_iam_crnn 目录下):
    python train.py --epochs 30
    python train.py --limit 2000 --epochs 2      # 小样本快速调试
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

from common import get_device, set_seed, get_iam_loaders, BLANK
from models import build_crnn, count_parameters, greedy_decode


# ============================ 评价指标 ============================

def edit_distance(a, b) -> int:
    """Levenshtein 编辑距离(插入/删除/替换各计 1)。"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # 删除
                           cur[j - 1] + 1,       # 插入
                           prev[j - 1] + (ca != cb)))  # 替换/相同
        prev = cur
    return prev[-1]


def compute_cer_wer(pred_texts, true_texts):
    """字符错误率 CER 与单词错误率 WER(编辑距离 / 参考总长度)。"""
    cer_num = sum(edit_distance(p, t) for p, t in zip(pred_texts, true_texts))
    cer_den = sum(len(t) for t in true_texts) or 1
    wer_num = sum(0 if p == t else 1 for p, t in zip(pred_texts, true_texts))
    return cer_num / cer_den, wer_num / (len(true_texts) or 1)


# ============================ 单轮训练 / 评估 ============================

def run_epoch(model, loader, criterion, device, vocab, optimizer=None, desc=""):
    """跑一个 epoch。optimizer 为 None 时为评估模式,额外统计 CER/WER。"""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_seen = 0.0, 0
    all_preds, all_trues = [], []

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
        for x, targets, target_lengths, texts in pbar:
            x = x.to(device)
            logits = model(x)                          # [T, N, C]
            T, N, C = logits.shape
            log_probs = logits.log_softmax(2)

            input_lengths = torch.full((N,), T, dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪:CTC + RNN 组合容易梯度爆炸,这一步很关键
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * N
            n_seen += N

            if not train_mode:
                preds = greedy_decode(log_probs, blank=BLANK)
                all_preds.extend(vocab.decode(p) for p in preds)
                all_trues.extend(texts)
            pbar.set_postfix(loss=f"{total_loss / max(n_seen, 1):.3f}")

    avg_loss = total_loss / max(n_seen, 1)
    if train_mode:
        return avg_loss, None
    cer, wer = compute_cer_wer(all_preds, all_trues)
    return avg_loss, (cer, wer, all_preds, all_trues)


def main():
    ap = argparse.ArgumentParser(description="第3周 CRNN+CTC 单词级识别")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn-layers", type=int, default=2)
    ap.add_argument("--augment", action="store_true", help="训练集启用轻量增强")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每个划分只取前 N 条(调试用)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    args = ap.parse_args()

    tag = f"crnn_h{args.hidden}" + ("_aug" if args.augment else "") + (f"_limit{args.limit}" if args.limit else "")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "week3", tag)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"设备={device}  模型={tag}")

    train_loader, val_loader, test_loader, vocab = get_iam_loaders(
        args.data_dir, batch_size=args.batch_size, augment=args.augment,
        num_workers=args.num_workers, seed=args.seed, limit=args.limit)
    print(f"划分大小 train/val/test = {len(train_loader.dataset)}/"
          f"{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"词表大小(含 blank)= {len(vocab)}")

    model = build_crnn(len(vocab), hidden_size=args.hidden, rnn_layers=args.rnn_layers).to(device)
    n_params = count_parameters(model)
    print(f"可训练参数量 = {n_params:,}")

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_wer": []}
    best_cer = float("inf")
    ckpt = os.path.join(out_dir, "best_model.pt")

    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, vocab,
                               optimizer, desc=f"train {epoch}/{args.epochs}")
        va_loss, (va_cer, va_wer, _, _) = run_epoch(model, val_loader, criterion,
                                                    device, vocab, desc=f"val   {epoch}/{args.epochs}")
        scheduler.step(va_cer)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_cer"].append(va_cer)
        history["val_wer"].append(va_wer)

        print(f"  epoch {epoch:2d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_CER={va_cer * 100:.2f}%  val_WER={va_wer * 100:.2f}%")

        if va_cer < best_cer:
            best_cer = va_cer
            torch.save(model.state_dict(), ckpt)
    train_time = time.perf_counter() - t_start

    # ---- 用最优权重在测试集上评估 ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, (te_cer, te_wer, preds, trues) = run_epoch(model, test_loader, criterion,
                                                  device, vocab, desc="test")
    print(f"\n最优验证 CER = {best_cer * 100:.2f}%")
    print(f"测试集 CER   = {te_cer * 100:.2f}%")
    print(f"测试集 WER   = {te_wer * 100:.2f}%")
    print(f"训练总耗时   = {train_time:.1f}s")

    # ---- 训练曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{tag} - CTC Loss")
    axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot([c * 100 for c in history["val_cer"]], label="CER (%)")
    axes[1].plot([w * 100 for w in history["val_wer"]], label="WER (%)")
    axes[1].set_title(f"{tag} - Val Error Rate")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)

    # ---- 保存结果 ----
    result = {
        "tag": tag, "model": "crnn", "hidden": args.hidden,
        "rnn_layers": args.rnn_layers, "augment": bool(args.augment),
        "limit": args.limit, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "params": n_params, "vocab_size": len(vocab),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
        "best_val_cer": best_cer, "test_cer": te_cer, "test_wer": te_wer,
        "train_time_sec": train_time, "history": history,
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 顺便存一批预测样例,供 evaluate.py / 报告使用
    with open(os.path.join(out_dir, "test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump([{"pred": p, "true": t} for p, t in zip(preds, trues)],
                  f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
