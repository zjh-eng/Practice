# 基于深度学习的手写文字识别系统 (HTR)

> 杭州师范大学人工智能实验室 · 2026 级研究生入学前预研任务
> 项目周期:4 周 —— 从零搭建一个「图片 → 文字」的手写文字识别系统。

## 一、项目目标

在一个月内完成一个可运行的手写文字识别系统,实现从图片输入到文字输出的完整流程。
例如输入一张手写单词 `Hello` 的图片,系统输出识别结果 `Hello`。

通过本项目掌握:Python 工程环境、PyTorch、数据集处理与模型训练、CNN/CRNN 经典结构、
Transformer OCR 技术、Git/GitHub 代码管理、AI 辅助编程、科研报告撰写。

## 二、四阶段路线图

| 阶段 | 时间 | 任务 | 目录 | 验收标准 |
|---|---|---|---|---|
| 1 | 第1周 | MNIST 数字识别(跑通训练流程) | `week1_mnist/` | 测试集 ≥ 98% |
| 2 | 第2周 | EMNIST 字符识别(MLP/CNN/ResNet18 对比) | `week2_emnist/` | 对比表(准确率/时间/参数量) |
| 3 | 第3周 | IAM 单词级识别(CRNN + CTC) | `week3_iam_crnn/` | 正确 + 失败案例展示 |
| 4 | 第4周 | Transformer OCR 调研(TrOCR/Donut 等) | `week4_transformer_ocr/` | 调研报告 |

## 三、环境配置

本机已验证可用的环境为 conda 环境 `tuduipytorch`(Python 3.13 + torch 2.9.1+cu130,GPU 可用)。

```bash
# 1. 激活环境
conda activate tuduipytorch

# 2. 验证 GPU 是否可用(应输出 True)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Windows 下 OMP 冲突说明**:若导入 torch 时报
> `libiomp5md.dll already initialized`,是 numpy(MKL)与 torch 的 OpenMP 运行库重复导致的,
> 本项目所有脚本已在顶部通过 `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` 规避。

若需从零建环境,可参考:

```bash
conda create -n htr python=3.11 -y
conda activate htr
# GPU 版 PyTorch(按你的 CUDA 版本到 https://pytorch.org 选安装命令)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 四、快速开始(第1周)

```bash
cd week1_mnist
python train.py            # 训练 + 画曲线 + 存模型
python evaluate.py         # 测试准确率 + 混淆矩阵 + 错样本分析
```

产物保存在 `outputs/week1/`:训练曲线、混淆矩阵、错样本图、模型权重。
数据自动下载到 `data/`(已 gitignore)。

## 五、目录结构

```
.
├── week1_mnist/            # 第1周:MNIST 数字识别
├── week2_emnist/           # 第2周:EMNIST 字符识别
├── week3_iam_crnn/         # 第3周:IAM 单词级识别(CRNN+CTC)
├── week4_transformer_ocr/  # 第4周:Transformer OCR 调研
├── docs/                   # 文献笔记、报告素材
├── outputs/                # 运行产物(已 gitignore)
├── data/                   # 数据集(已 gitignore)
├── requirements.txt        # 依赖清单
└── .gitignore
```

## 六、交付物清单

1. ✅ GitHub 仓库(源代码 + README + 环境配置说明)
2. 项目技术报告(PDF,10~15 页)
3. 项目汇报 PPT
4. 演示视频(≤5 分钟)
5. 文献阅读笔记(CRNN、TrOCR、Donut)

## 七、AI 辅助开发说明(任务书要求)

本项目使用 **Claude Code** 辅助开发,使用情况(工具、完成的工作、AI 代码占比估计、心得)
将随项目推进记录于各周 `README` 及最终报告中。
