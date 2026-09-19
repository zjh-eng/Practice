"""第2周公共工具:EMNIST 数据加载、数据增强、设备与随机种子。

EMNIST 相比 MNIST 有两个必须注意的坑:
  1. 图像方向:原始 idx 文件以「转置」形式存储,torchvision 直接读取会得到
     旋转 90° 的图,需要交换 H/W 两个维度还原(见 TRANSPOSE_EMNIST)。
  2. 标签偏移:只有 letters 子集的标签从 1 开始(0 是占位),其余子集正常。
     本项目默认使用 balanced 子集(47 类,数字+大写字母+部分小写字母),标签无偏移。
"""
import os

# OMP 冲突规避(需在 import torch 之前)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 项目根目录(由脚本位置推导,不受 PyCharm「工作目录」设置影响)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---- EMNIST 配置 ----
DEFAULT_SPLIT = "balanced"          # 47 类:0-9、A-Z、以及 11 个未合并的小写字母
TRANSPOSE_EMNIST = True             # 修正图像方向(由 check_data.py 实测验证)
EMNIST_MEAN, EMNIST_STD = 0.1751, 0.3332   # 由 train.py 首次运行时实测校准


def get_device() -> torch.device:
    """返回可用计算设备:优先 GPU,否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """固定随机种子,保证实验可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _orient(x: torch.Tensor) -> torch.Tensor:
    """把 ToTensor 之后的 [C, H, W] 交换 H/W,还原 EMNIST 的正确方向。"""
    return x.transpose(1, 2).contiguous() if TRANSPOSE_EMNIST else x


def build_transform(augment: bool = False):
    """构造 transform。augment=True 时加入数据增强(仅用于训练集)。

    增强策略:
      - RandomAffine:轻微旋转 ±10°、平移 8%、缩放 90%~110%,
        模拟手写时角度和位置的天然差异;
      - RandomErasing:随机遮挡一小块,逼模型别依赖单一局部笔画。
    注意:增强必须在 Normalize 之前(在像素空间做几何变换才有物理意义)。
    """
    ops = [transforms.ToTensor(), transforms.Lambda(_orient)]
    if augment:
        ops += [
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), value=0.0),
        ]
    ops.append(transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)))
    return transforms.Compose(ops)


class _IndexedDataset(Dataset):
    """在「原始数据集 + 指定索引 + 指定 transform」上取数据。

    这样可以让训练集和验证集共用同一份底层数据(省内存),
    却使用不同的 transform(训练集增强、验证集不增强)。
    """

    def __init__(self, base: Dataset, indices, transform=None):
        self.base = base
        self.transform = transform
        self.indices = list(range(len(base))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_emnist_loaders(data_dir: str, split: str = DEFAULT_SPLIT, batch_size: int = 128,
                       augment: bool = False, val_ratio: float = 0.1, seed: int = 42,
                       num_workers: int = 0):
    """构造 EMNIST 的 train / val / test 三个 DataLoader。

    返回:(train_loader, val_loader, test_loader, class_names)
    - 训练集使用带增强(或不带)的 transform;
    - 验证集从训练集中按 val_ratio 切出,但**不加增强**(否则评估会失真);
    - 测试集独立,用于最终评估。
    Windows 下 num_workers 默认 0,避免多进程加载的 spawn 报错。
    """
    raw_train = datasets.EMNIST(data_dir, split=split, train=True,
                                download=True, transform=None)
    raw_test = datasets.EMNIST(data_dir, split=split, train=False,
                               download=True, transform=None)

    n_val = int(len(raw_train) * val_ratio)
    n_train = len(raw_train) - n_val
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(raw_train), generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = _IndexedDataset(raw_train, train_idx, build_transform(augment))
    val_ds = _IndexedDataset(raw_train, val_idx, build_transform(False))
    test_ds = _IndexedDataset(raw_test, None, build_transform(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader, raw_train.classes
