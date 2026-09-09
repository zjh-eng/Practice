"""第1周公共工具:设备选择、随机种子、MNIST 数据加载。"""
import os
import random

# Windows 下 torch 与 numpy(MKL)同时加载 OpenMP 运行库会冲突,
# 报 "libiomp5md.dll already initialized" 错误。
# 必须在 import torch 之前设置该环境变量规避(详见根目录 README)。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """返回可用计算设备:优先 GPU,否则 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """固定随机种子,保证实验可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_mnist_loaders(data_dir: str, batch_size: int = 64):
    """构造 MNIST 的 train / val / test 三个 DataLoader。

    训练集按 9:1 切出验证集;测试集保持独立(用于最终评估)。
    Windows 下 num_workers 默认 0,避免多进程数据加载的 spawn 报错,
    MNIST 数据量小,CPU 加载足够快。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                        # [0, 255] -> [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 均值 / 标准差
    ])

    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    n_train = int(len(full_train) * 0.9)
    n_val = len(full_train) - n_train
    train, val = torch.utils.data.random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
