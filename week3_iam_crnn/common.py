"""第3周公共工具:IAM 单词级数据加载、词表构建、CTC 编码。

IAM 数据布局(解压后为平铺结构,没有 words/、ascii/ 外层目录):
    data/IAM/
    ├── a01/a01-000u/a01-000u-00-00.png      # 单词图像,按 写手/表单 分层
    ├── a02/ ... l07/
    ├── words.txt                            # 标签:每行一个单词
    └── splits/{train,validation,test}.uttlist   # 按「表单」划分

words.txt 每行 9 个字段:
    单词ID  状态  灰度  x y w h  词性  转写文本
    a01-000u-00-00 ok 154 408 768 27 51 AT A
其中状态 ok 表示切分正确(er 表示切分可能有误),本项目只用 ok 的样本。
"""
import os

# OMP 冲突规避(需在 import torch 之前)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 项目根目录(由脚本位置推导,不受 PyCharm「工作目录」设置影响)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---- 图像统一尺寸:等比缩放到高 32,宽度不足补白边 ----
IMAGE_HEIGHT = 32
MAX_WIDTH = 192

# ---- CTC 约定:索引 0 固定为 blank(空白符) ----
BLANK = 0


def get_device() -> torch.device:
    """返回可用计算设备:优先 GPU,否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """固定随机种子,保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================ 数据解析 ============================

def parse_words_txt(words_txt: str):
    """解析 words.txt,返回 [(word_id, transcription), ...],只保留 ok 的样本。"""
    out = []
    with open(words_txt, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:          # 字段不全的行跳过
                continue
            word_id, status = parts[0], parts[1]
            if status != "ok":          # 只看切分正确的单词
                continue
            out.append((word_id, parts[-1]))   # 转写文本是最后一个字段
    return out


def form_id_of(word_id: str) -> str:
    """单词 ID -> 所属表单 ID。例:a01-000u-00-00 -> a01-000u"""
    return "-".join(word_id.split("-")[:2])


def image_path_of(iam_dir: str, word_id: str) -> str:
    """单词 ID -> 图像路径。例:a01-000u-00-00 ->
    data/IAM/a01/a01-000u/a01-000u-00-00.png"""
    form = form_id_of(word_id)
    writer = form.split("-")[0]
    return os.path.join(iam_dir, writer, form, word_id + ".png")


def load_splits(splits_dir: str):
    """读取 Aachen 划分,返回 {'train': set(form_id), 'validation': ..., 'test': ...}。"""
    splits = {}
    for name in ("train", "validation", "test"):
        with open(os.path.join(splits_dir, name + ".uttlist"), encoding="utf-8") as f:
            splits[name] = {l.strip() for l in f if l.strip()}
    return splits


# ============================ 词表与编码 ============================

class Vocab:
    """字符级词表。索引 0 留给 CTC 的 blank,真实字符从 1 开始。"""

    def __init__(self, chars):
        self.chars = list(chars)
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars) + 1        # +1 是 blank

    def encode(self, text: str):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices):
        return "".join(self.idx2char.get(i, "") for i in indices)

    @classmethod
    def from_texts(cls, texts):
        chars = sorted({c for t in texts for c in t})
        return cls(chars)


# ============================ 图像变换 ============================

def resize_pad(img: Image.Image) -> Image.Image:
    """等比缩放到高度 IMAGE_HEIGHT;宽度超限则压缩,不足则右侧补白边。

    白边与 IAM 图像背景一致(背景为白 255、字为黑),因此补边不会引入
    额外的视觉噪声,模型只需学会「白边处输出 blank」。
    """
    w, h = img.size
    new_w = max(1, int(round(w * IMAGE_HEIGHT / h)))
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH
    img = img.resize((new_w, IMAGE_HEIGHT), Image.BILINEAR)
    canvas = Image.new("L", (MAX_WIDTH, IMAGE_HEIGHT), color=255)
    canvas.paste(img, (0, 0))
    return canvas


def build_transform(augment: bool = False):
    """构造 transform。augment=True 时加入轻量增强(仅训练集)。"""
    ops = []
    if augment:
        ops.append(transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)))
    ops.append(transforms.ToTensor())                       # -> [1, 32, 192], 取值 [0,1]
    ops.append(transforms.Normalize((0.9238,), (0.1848,)))   # 训练集 3000 张实测定标
    return transforms.Compose(ops)


# ============================ Dataset ============================

class IAMWordDataset(Dataset):
    """IAM 单词级数据集:返回 (图像张量, 编码后的标签, 原始文本)。"""

    def __init__(self, samples, vocab: Vocab, augment: bool = False):
        self.samples = samples          # [(image_path, text), ...]
        self.vocab = vocab
        self.transform = build_transform(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # IAM 压缩包里有极少量 0 字节的损坏图片,直接打开会抛
        # UnidentifiedImageError 中断整个训练。这里向后顺延到最近一个
        # 可用的样本,保证训练不中断(顺延的样本仍是真实样本,不影响正确性)。
        for offset in range(8):
            path, text = self.samples[(i + offset) % len(self.samples)]
            try:
                img = Image.open(path).convert("L")
                img.load()                       # 触发真正解码,坏图在此暴露
                break
            except Exception:
                continue
        else:
            # 极端情况:连续 8 个都损坏,返回全白图 + 空标签(几乎不会发生)
            img = Image.new("L", (MAX_WIDTH, IMAGE_HEIGHT), color=255)
            text = ""

        img = resize_pad(img)
        x = self.transform(img)
        y = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return x, y, text


def collate_ctc(batch):
    """CTC 专用 batch 组装。

    - 图像尺寸已统一,可直接 stack;
    - 标签长度不一,按 CTC 要求拼接成一维长张量,并记录每条的长度。
    """
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    texts = [b[2] for b in batch]
    return imgs, targets, target_lengths, texts


def get_iam_loaders(data_dir: str, batch_size: int = 64, augment: bool = False,
                    num_workers: int = 0, seed: int = 42, limit: int = 0):
    """构造 IAM 单词级的 train / val / test 三个 DataLoader。

    参数 limit > 0 时只取每个划分的前 limit 条(用于快速调试)。
    返回:(train_loader, val_loader, test_loader, vocab)
    """
    iam_dir = os.path.join(data_dir, "IAM")
    words_txt = os.path.join(iam_dir, "words.txt")
    splits_dir = os.path.join(iam_dir, "splits")

    pairs = parse_words_txt(words_txt)
    splits = load_splits(splits_dir)

    # 按「表单」把单词分到三个划分里
    bucket = {"train": [], "validation": [], "test": []}
    for word_id, text in pairs:
        form = form_id_of(word_id)
        path = image_path_of(iam_dir, word_id)
        if not os.path.exists(path):        # 极少数缺失文件直接跳过
            continue
        for name in ("train", "validation", "test"):
            if form in splits[name]:
                bucket[name].append((path, text))
                break

    if limit:
        for k in bucket:
            bucket[k] = bucket[k][:limit]

    # 词表只用训练集构建(避免验证/测试集信息泄漏)
    vocab = Vocab.from_texts([t for _, t in bucket["train"]])

    g = torch.Generator().manual_seed(seed)
    train_ds = IAMWordDataset(bucket["train"], vocab, augment)
    val_ds = IAMWordDataset(bucket["validation"], vocab, False)
    test_ds = IAMWordDataset(bucket["test"], vocab, False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_ctc)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_ctc)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_ctc)
    return train_loader, val_loader, test_loader, vocab
