# 第1周 · MNIST 数字识别

## 目标
熟悉 PyTorch 完整训练流程,完成基础图像分类任务,测试集准确率 ≥ 98%。

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | CNN 模型定义(两层卷积 + 两层全连接) |
| `common.py` | 设备选择、随机种子、MNIST 数据加载 |
| `train.py` | 训练 + 验证,画 loss/acc 曲线,保存最优模型 |
| `evaluate.py` | 测试集评估,输出混淆矩阵和错样本可视化 |

## 运行

```bash
# 在 week1_mnist 目录下
conda activate tuduipytorch
python train.py                     # 默认 5 epoch
python train.py --epochs 10         # 自定义轮数
python evaluate.py                  # 需要先跑完 train.py
```

## 任务书要求对照

| 任务要求 | 对应 |
|---|---|
| 下载并加载 MNIST | `common.py` 中 `torchvision.datasets.MNIST` |
| CNN 数字分类模型 | `model.py` |
| 训练、验证、测试 | `train.py` + `evaluate.py` |
| 损失/准确率曲线 | `train.py` → `training_curves.png` |
| 混淆矩阵 | `evaluate.py` → `confusion_matrix.png` |
| 错误样本分析 | `evaluate.py` → `error_analysis.png` |

## 模型结构

```
Input [1, 28, 28]
  -> Conv(1->32, 3x3) -> ReLU -> MaxPool(2)   # 28x28 -> 14x14
  -> Conv(32->64, 3x3) -> ReLU -> MaxPool(2)  # 14x14 -> 7x7
  -> Flatten -> Linear(64*7*7 -> 128) -> ReLU -> Dropout(0.5)
  -> Linear(128 -> 10)
```

## 结果记录

- 测试集准确率:`99.16%`
- 主要错样本类型:以形近数字混淆为主 —— 4→9(11次)、3→5(8次)、2↔7(6/5次);测试集共 84 个错样本
