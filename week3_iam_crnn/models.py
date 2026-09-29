"""第3周模型:CRNN(Convolutional Recurrent Neural Network)。

整体流程(任务书要求理解的主线):
    Image -> CNN -> Feature Sequence -> BiLSTM -> CTC Decoder -> Text

设计要点:
  1. CNN 负责从图像中提取特征,并在**宽度方向**保留序列信息
     (只在高度方向下采样,宽度方向保留,这样每个时间步对应图像的一小段);
  2. 把 CNN 特征图按宽度切成若干时间步,展平成序列送入 BiLSTM,
     双向 LSTM 同时利用左右文脉信息;
  3. 输出每个时间步在所有字符上的分数,交给 CTC Loss 处理「对齐」问题。
"""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """输入 [N, 1, 32, W] -> 输出 [T, N, num_classes] 的逐时间步 logits。

    T = W / 4(W 为图像宽度,本项目固定 192,故 T = 48)。
    """

    def __init__(self, num_classes: int, hidden_size: int = 256, rnn_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        # ---------- ① CNN 特征提取 ----------
        # 高度方向两次 2 倍下采样:32 -> 16 -> 8
        # 宽度方向两次 2 倍下采样:W  -> W/2 -> W/4 (后续只在高度方向池化,保住序列长度)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # 32 x W   -> 16 x W/2

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # 16 x W/2 -> 8  x W/4

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                                 # 8 x W/4 -> 4 x W/4

            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                                 # 4 x W/4 -> 2 x W/4
        )

        # ---------- ② 特征序列化 ----------
        # 特征图 [N, 256, 2, T] -> 每个时间步取 256*2=512 维向量 -> [T, N, 512]
        self.feature_dim = 256 * 2

        # ---------- ③ 双向 LSTM ----------
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=False,       # 输入为 [T, N, C]
        )

        # ---------- ④ 逐时间步分类 ----------
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [N, 1, 32, W]
        conv = self.cnn(x)                        # [N, 256, 2, T]
        n, c, h, t = conv.size()
        conv = conv.permute(3, 0, 1, 2)           # [T, N, 256, 2]
        conv = conv.reshape(t, n, c * h)          # [T, N, 512]

        rnn_out, _ = self.rnn(conv)               # [T, N, 2*hidden]
        return self.fc(rnn_out)                   # [T, N, num_classes]


def build_crnn(num_classes: int, hidden_size: int = 256, rnn_layers: int = 2):
    return CRNN(num_classes, hidden_size=hidden_size, rnn_layers=rnn_layers)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================ CTC 解码 ============================

def greedy_decode(log_probs: torch.Tensor, blank: int = 0):
    """贪心解码:每个时间步取最大概率的字符,再去重复、去 blank。

    参数 log_probs: [T, N, C] 的 log 概率(或任意单调可比的分数)
    返回:长度为 N 的字符串列表
    """
    indices = log_probs.argmax(dim=2)             # [T, N]
    indices = indices.transpose(0, 1).cpu().numpy()   # [N, T]

    results = []
    for seq in indices:
        out, prev = [], None
        for idx in seq:
            idx = int(idx)
            if idx != prev and idx != blank:      # 去掉连续重复 与 blank
                out.append(idx)
            prev = idx
        results.append(out)
    return results
