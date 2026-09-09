"""CNN 数字分类模型 —— 第1周 MNIST。

一个「两层卷积 + 两层全连接」的小型 CNN,
足以在 MNIST 测试集上稳定达到 98% 以上的准确率。
"""
from torch import nn


class MNIST_CNN(nn.Module):
    """输入 [N, 1, 28, 28] -> 输出 [N, num_classes] 的 logits。

    结构:
        Conv(1->32, 3x3, pad=1) -> ReLU -> MaxPool(2)   # 28 -> 14
        Conv(32->64, 3x3, pad=1) -> ReLU -> MaxPool(2)  # 14 -> 7
        Flatten -> Linear(64*7*7 -> 128) -> ReLU -> Dropout
                -> Linear(128 -> num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
