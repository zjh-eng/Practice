#!/usr/bin/env bash
# 第2周:一键跑完全部对比实验,并生成对比表与对比图。
# 用法: bash run_all.sh
set -e

PY="E:/Anaconda/envs/tuduipytorch/python.exe"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8

EPOCHS=${EPOCHS:-15}

echo "########## 1/5 MLP(基线,无增强) ##########"
"$PY" train.py --model mlp --epochs "$EPOCHS"

echo "########## 2/5 CNN(无增强) ##########"
"$PY" train.py --model cnn --epochs "$EPOCHS"

echo "########## 3/5 CNN + 数据增强 ##########"
"$PY" train.py --model cnn --epochs "$EPOCHS" --augment

echo "########## 4/5 ResNet18(无增强) ##########"
"$PY" train.py --model resnet18 --epochs "$EPOCHS"

echo "########## 5/5 汇总对比 ##########"
"$PY" compare.py

echo "ALL_DONE"
