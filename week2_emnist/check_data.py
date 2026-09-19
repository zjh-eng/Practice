# -*- coding: utf-8 -*-
"""数据校验脚本:用 ASCII 图确认 EMNIST 图像方向与标签映射是否正确。

EMNIST 的原始 idx 文件以「转置」形式存储,如果不在 transform 里交换 H/W,
模型看到的将是旋转 90° 的字。本脚本把同一个样本「转置前/转置后」都画成
字符画打印出来,肉眼即可判断哪种方向才是对的。

用法(在 week2_emnist 目录下):
    python check_data.py
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from torchvision import datasets

from common import PROJECT_ROOT, DEFAULT_SPLIT

RAMP = " .:-=+*#%@"  # 由暗到亮


def ascii_art(img: np.ndarray, width: int = 28) -> str:
    """把 [H, W] 的灰度图(0~255)转成字符画。"""
    lines = []
    for row in img:
        line = "".join(RAMP[min(int(v) * len(RAMP) // 256, len(RAMP) - 1)] for v in row)
        lines.append(line)
    return "\n".join(lines)


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    ds = datasets.EMNIST(data_dir, split=DEFAULT_SPLIT, train=True,
                         download=True, transform=None)

    print(f"数据集: EMNIST / split={DEFAULT_SPLIT}")
    print(f"训练集样本数: {len(ds)}")
    print(f"类别数: {len(ds.classes)}")
    print(f"类别列表: {''.join(ds.classes)}")

    target = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    idx = next(i for i, (_, y) in enumerate(ds) if y == target)
    img, label = ds[idx]
    arr = np.array(img)

    print(f"\n取样: index={idx}, label={label} (类别 '{ds.classes[label]}'), shape={arr.shape}")

    print("\n" + "=" * 60)
    print("【A】不做转置(transform=None 时 torchvision 直接读出的样子)")
    print("=" * 60)
    print(ascii_art(arr))

    print("\n" + "=" * 60)
    print("【B】交换 H/W 转置后(项目采用的方式)")
    print("=" * 60)
    print(ascii_art(arr.T))

    print("\n" + "=" * 60)
    print("判断方法:哪个方向能看出正常的字形,就用哪个。")
    print("=" * 60)


if __name__ == "__main__":
    main()
