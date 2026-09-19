"""第2周 EMNIST 字符识别的三种模型:MLP / CNN / ResNet18。

三个模型接收相同输入 [N, 1, 28, 28],输出 [N, num_classes] 的 logits,
便于在统一训练流程下做公平对比。
"""
import torch.nn as nn
from torchvision.models import resnet18


class MLP(nn.Module):
    """多层感知机基线:直接把图像展平成向量,不做任何局部特征提取。

    作用:作为「最弱基线」,用来体现卷积结构带来的提升。
    """

    def __init__(self, num_classes: int = 47, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EMNIST_CNN(nn.Module):
    """三层卷积 CNN(带 BatchNorm),在 MNIST 版 CNN 基础上加深加宽。

    结构:
        Conv(1->32)   + BN + ReLU + MaxPool   # 28 -> 14
        Conv(32->64)  + BN + ReLU + MaxPool   # 14 -> 7
        Conv(64->128) + BN + ReLU + MaxPool   # 7  -> 3
        Flatten -> FC(1152->256) -> ReLU -> Dropout -> FC(256->num_classes)
    """

    def __init__(self, num_classes: int = 47, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 47) -> nn.Module:
    """适配 28x28 单通道输入的 ResNet18。

    原始 ResNet18 为 ImageNet(224x224 三通道)设计,直接用在 28x28 上会因
    首层 7x7 大卷积 + 4 倍下采样而把特征图压到 1x1。这里做两处标准改造:
      1. 首层换成 3x3、stride=1、输入通道改为 1;
      2. 去掉紧跟首层的 MaxPool。
    这样 28x28 输入经过 4 个 stage 后仍保留 4x4 空间信息,并复用 ImageNet 的结构设计。
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# 模型名 -> 构造函数,供 train.py 按名字取用
MODEL_ZOO = {
    "mlp": MLP,
    "cnn": EMNIST_CNN,
    "resnet18": build_resnet18,
}


def build_model(name: str, num_classes: int = 47) -> nn.Module:
    if name not in MODEL_ZOO:
        raise ValueError(f"未知模型 '{name}',可选:{list(MODEL_ZOO)}")
    return MODEL_ZOO[name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量(用于对比表)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
