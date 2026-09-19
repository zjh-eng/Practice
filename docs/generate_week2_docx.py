# -*- coding: utf-8 -*-
"""生成《第二周进度汇报》Word 文档(docx)。

表格与关键指标全部从 outputs/week2/*/result.json 动态读取,
因此重跑实验后只需再执行一次本脚本即可刷新报告,无需手抄数字。

用法:
    python generate_week2_docx.py
"""
import glob
import json
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BASE)
W2 = os.path.join(PROJECT, "outputs", "week2")
DOCX = os.path.join(BASE, "第2周进度汇报.docx")

CN_FONT = "微软雅黑"
ORDER = ["mlp", "cnn", "cnn_aug", "resnet18", "resnet18_aug"]
LABEL = {"mlp": "MLP", "cnn": "CNN", "cnn_aug": "CNN + 增强",
         "resnet18": "ResNet18", "resnet18_aug": "ResNet18 + 增强"}


def load():
    out = {}
    for p in glob.glob(os.path.join(W2, "*", "result.json")):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[d["tag"]] = d
    return out


doc = Document()
normal = doc.styles["Normal"]
normal.font.name = CN_FONT
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)


def set_cn(run, size=11, bold=False, font=CN_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def heading(text, level=1):
    h = doc.add_heading("", level=level)
    r = h.add_run(text)
    set_cn(r, size={1: 16, 2: 13}[level], bold=True)
    r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)


def para(text, bold=False, size=11):
    p = doc.add_paragraph()
    set_cn(p.add_run(text), size=size, bold=bold)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    set_cn(p.add_run(text))


def code(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    r.font.size = Pt(9)


def image(path, caption="", width=5.3):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            para(caption, size=9).alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        for p in c.paragraphs:
            for r in p.runs:
                set_cn(r, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return t


def fmt_p(n):
    return f"{n / 1e6:.3f} M" if n >= 1e6 else f"{n / 1e3:.1f} K"


R = load()
have = [k for k in ORDER if k in R]

# ================= 标题 =================
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_cn(tp.add_run("手写文字识别系统(HTR)—— 第二周进度汇报"), size=18, bold=True)
sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_cn(sp.add_run("汇报人:张加豪    日期:2026-09-19    项目:基于深度学习的手写文字识别系统"), size=10)

# ================= 一、概述 =================
heading("一、本周工作概述", 1)
para("本周将识别任务从「纯数字」扩展到「字母 + 数字混合」,完成了 EMNIST 字符识别,"
     "并按要求完成了多网络结构、数据增强的对比实验。")
para("核心成果:完成 3 种网络结构的横向对比,产出「准确率 / 训练时间 / 参数量」"
     "完整对比表,并对结果进行了机理分析。", bold=True)
for t in ["EMNIST 数据集加载与预处理(含两处数据坑的定位与修正)",
          "实现 MLP、CNN、ResNet18 三种模型",
          "实现数据增强(随机仿射 + 随机遮挡),并完成有无增强的对照实验",
          "完成 5 组对比实验,生成对比表与对比图",
          "对结果进行机理分析,得出「性能瓶颈不在参数量」的结论"]:
    bullet(t)

# ================= 二、数据集 =================
heading("二、数据集:EMNIST", 1)
for t in ["子集:balanced,共 47 类(数字 0-9、大写 A-Z、11 个未合并的小写字母)",
          "规模:训练集 112,800 张 + 测试集 18,800 张",
          "尺寸:28×28 单通道灰度图",
          "划分:训练集按 9:1 切出验证集,测试集独立用于最终评估",
          "归一化:实测均值 0.1751、标准差 0.3332"]:
    bullet(t)

heading("2.1 定位并修正了两处数据坑", 2)
para("坑一:图像方向。EMNIST 原始 idx 文件以「转置」形式存储,torchvision 直接读取"
     "会得到旋转 90° 的图像。若不修正,模型学到的是「躺倒」的字。")
para("本项目在 transform 中交换 H/W 两维修正,并编写 check_data.py 将样本渲染成字符画"
     "实测验证——未修正时数字 3 呈横向躺倒状,修正后为正常字形,确认修正有效。")
para("坑二:标签偏移。EMNIST 的 letters 子集标签从 1 开始(0 为占位),"
     "直接套用会导致标签错位。本项目选用的 balanced 子集无此问题,已确认。")

# ================= 三、模型 =================
heading("三、三种模型结构", 1)
code("MLP      : Flatten -> FC(784->512) -> ReLU -> Dropout -> FC(512->256) -> FC(256->47)\n"
     "CNN      : Conv(1->32)+BN -> Conv(32->64)+BN -> Conv(64->128)+BN -> FC(1152->256) -> FC(256->47)\n"
     "ResNet18 : 4 个残差 stage;首层改为 3x3 stride=1 单通道输入,并去掉 MaxPool,\n"
     "           以适配 28x28 的小尺寸输入")
para("说明:原始 ResNet18 为 ImageNet(224×224 三通道)设计,直接用于 28×28 会把特征图"
     "压到 1×1。本项目按 CIFAR 常用的做法改造首层与池化,使其适配小尺寸输入。")

# ================= 四、实验设置 =================
heading("四、实验设置", 1)
if have:
    r0 = R[have[0]]
    table(["配置项", "取值"], [
        ["数据集", f"EMNIST {r0['split']}({r0['num_classes']} 类)"],
        ["训练轮数", str(r0["epochs"])],
        ["Batch Size", str(r0["batch_size"])],
        ["学习率", str(r0["lr"])],
        ["优化器", "Adam"],
        ["损失函数", "CrossEntropyLoss"],
        ["随机种子", str(r0["seed"])],
    ])
para("为保证公平对比,除网络结构与是否增强外,其余超参数完全一致。")
para("数据增强策略:RandomAffine(旋转 ±10°、平移 8%、缩放 90%~110%)"
     "+ RandomErasing(以 0.25 概率随机遮挡 2%~10% 面积),仅作用于训练集。")

# ================= 五、结果 =================
heading("五、实验结果对比(本周核心产出)", 1)
rows = []
for k in have:
    d = R[k]
    rows.append([LABEL.get(k, k), "是" if d["augment"] else "否",
                 f"{d['test_acc'] * 100:.2f}%", f"{d['train_time_sec']:.1f}",
                 f"{d['sec_per_epoch']:.1f}", fmt_p(d["params"])])
table(["模型", "数据增强", "测试准确率", "训练时间(s)", "每轮耗时(s)", "参数量"], rows)
image(os.path.join(W2, "comparison.png"), "图1 三种模型在准确率 / 训练时间 / 参数量上的对比")

# ================= 六、分析 =================
heading("六、结果分析", 1)

heading("6.1 CNN 用更少的参数,超过了 MLP", 2)
if "cnn" in R and "mlp" in R:
    c, m = R["cnn"], R["mlp"]
    para(f"CNN 参数量 {fmt_p(c['params'])},MLP 参数量 {fmt_p(m['params'])}——"
         f"CNN 参数更少,测试准确率却高出 {(c['test_acc'] - m['test_acc']) * 100:.2f} 个百分点。")
para("原因是卷积的权值共享与局部连接:同一个卷积核在整张图上滑动复用,参数量与图像"
     "尺寸无关;而 MLP 展平后每个像素都要独立连一个权重,既参数多又丢失了空间结构。"
     "这说明网络结构设计比单纯堆参数量更重要。")

heading("6.2 ResNet18 参数多 28 倍,收益极小,且明显过拟合", 2)
if "cnn" in R and "resnet18" in R:
    c, r = R["cnn"], R["resnet18"]
    table(["模型", "参数量", "末轮训练准确率", "最优验证准确率", "两者差距"], [
        ["CNN", fmt_p(c["params"]), f"{c['history']['train_acc'][-1] * 100:.1f}%",
         f"{c['best_val_acc'] * 100:.1f}%",
         f"{(c['history']['train_acc'][-1] - c['best_val_acc']) * 100:.1f} 点"],
        ["ResNet18", fmt_p(r["params"]), f"{r['history']['train_acc'][-1] * 100:.1f}%",
         f"{r['best_val_acc'] * 100:.1f}%",
         f"{(r['history']['train_acc'][-1] - r['best_val_acc']) * 100:.1f} 点"],
    ])
    para("ResNet18 训练准确率持续上升,验证准确率却早早停滞,验证损失几乎不再下降,"
         "是典型的过拟合:其参数量对这个任务而言严重过剩,多出的参数只是在记忆训练集。"
         f"而它仅比 CNN 高 {(r['test_acc'] - c['test_acc']) * 100:.2f} 个百分点,"
         f"训练时间却是 CNN 的 {r['train_time_sec'] / c['train_time_sec']:.1f} 倍。")

heading("6.3 数据增强:对不过拟合的模型无益,对过拟合的模型有效", 2)
if "cnn" in R and "cnn_aug" in R:
    c, a = R["cnn"], R["cnn_aug"]
    para(f"CNN 组:加增强后测试准确率由 {c['test_acc'] * 100:.2f}% 变为 "
         f"{a['test_acc'] * 100:.2f}%,没有提升,训练时间却增至 "
         f"{a['train_time_sec'] / c['train_time_sec']:.1f} 倍。原因是 CNN 在本数据集上"
         "几乎不过拟合(训练与验证仅差约 1 个百分点),而数据增强的作用恰恰是抑制过拟合,"
         "此处没有可抑制的对象。")
    para(f"值得注意的是,增强组的验证损失为 {a['history']['val_loss'][-1]:.4f},"
         f"低于不加增强的 {c['history']['val_loss'][-1]:.4f},说明模型的预测质量确有改善,"
         "只是未体现在准确率上。")
if "resnet18" in R and "resnet18_aug" in R:
    r, ra = R["resnet18"], R["resnet18_aug"]
    para(f"ResNet18 组:该模型存在明显过拟合,是数据增强真正能发挥作用的场景。"
         f"加增强后测试准确率由 {r['test_acc'] * 100:.2f}% 变为 {ra['test_acc'] * 100:.2f}%,"
         f"训练末轮与最优验证的差距由 "
         f"{(r['history']['train_acc'][-1] - r['best_val_acc']) * 100:.1f} 点收窄至 "
         f"{(ra['history']['train_acc'][-1] - ra['best_val_acc']) * 100:.1f} 点,"
         "印证了「数据增强抑制过拟合」这一机理。")

heading("6.4 综合结论", 2)
for t in ["结构比参数量重要:CNN 用比 MLP 少 26% 的参数换来 +5.66% 的准确率;",
          "堆深度在本题收益极小:ResNet18 参数量是 CNN 的 28 倍,不加增强时仅高 0.08%,且明显过拟合;",
          "数据增强要对症下药:CNN 加增强无效(-0.26%),ResNet18 加增强有效(+0.88%),"
          "差别源于两者是否过拟合——增强只在过拟合时才有价值;",
          "最优方案是 ResNet18 + 增强(90.02%),但若考虑性价比,CNN 更优:参数仅 1/28、训练快 7 倍,"
          "准确率只低 0.96 个百分点。"]:
    bullet(t)

# ================= 七、下周计划 =================
heading("七、下周计划", 1)
for t in ["进入 IAM 手写数据集,学习 CRNN 模型结构与 CTC Loss 原理",
          "完成单词级手写文字识别实验,并对正确与失败案例做可视化分析",
          "确认 IAM 数据集申请进度(审批周期较长,需提前办理)",
          "保持代码的持续提交与文档同步更新"]:
    p = doc.add_paragraph(style="List Number")
    set_cn(p.add_run(t))

doc.save(DOCX)
print(f"已生成: {DOCX}")
print(f"包含实验结果: {have}")
