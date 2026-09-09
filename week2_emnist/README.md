# 第2周 · EMNIST 字符识别

## 目标
从数字识别扩展到「字母 + 数字」混合识别,并做多模型对比实验。

## 计划任务

1. 学习 EMNIST 数据集结构(注意 `split` 与标签偏移问题)
2. 实现字符分类模型
3. 使用数据增强(`torchvision.transforms`)提升泛化能力
4. 对比至少两种网络结构:MLP / CNN / ResNet18
5. 分析实验结果

## 待完成

- [ ] 数据加载与预处理
- [ ] MLP / CNN / ResNet18 三种模型实现
- [ ] 数据增强实验
- [ ] 对比表(准确率 / 训练时间 / 参数量)

## 参考资料

- EMNIST 说明:https://www.nist.gov/itl/products-and-services/emnist-dataset
- EMNIST 论文:https://arxiv.org/abs/1702.05373
- PyTorch Transforms:https://pytorch.org/vision/stable/transforms.html
- ResNet18:https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
