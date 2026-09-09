# 第3周 · IAM 单词级手写文字识别(CRNN + CTC)

## 目标
进入真正的 HTR 任务,完成单词级手写识别。

整体流程:
```
Image -> CNN -> Feature Sequence -> BiLSTM -> CTC Decoder -> Text
```

## 计划任务

1. 阅读 IAM 数据集文档(**注意:IAM 需邮件申请,建议第1周就发出申请**)
2. 学习 CRNN 模型结构
3. 理解 CTC Loss 原理(不定长序列对齐)
4. 完成单词级识别实验
5. 可视化正确 / 失败案例

## 待完成

- [ ] IAM 数据申请与解析(单词切分、标签构建)
- [ ] CRNN 模型实现(CNN + BiLSTM)
- [ ] `torch.nn.CTCLoss` 训练
- [ ] CTC 解码(贪心 / beam search)
- [ ] 正确与失败案例分析

## 参考资料

- CRNN 论文:https://arxiv.org/abs/1507.05717
- CTC 原始论文:https://www.cs.toronto.edu/~graves/icml_2006.pdf
- PyTorch CTCLoss:https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
