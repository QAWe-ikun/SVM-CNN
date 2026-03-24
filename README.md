# CIFAR-10 图像分类：SVM vs CNN 对比分析

本项目使用 Python + OpenCV 实现 Sobel 边缘检测算法，并分别使用 **SVM** 和 **CNN（VGG 风格）** 实现 CIFAR-10 图像分类，对比两种方法的准确率和计算效率。

## 📁 项目结构

```
Task1/
├── data/                       # 数据集目录
├── results/                    # 结果输出目录
├── models/                     # 模型保存目录
├── src/                        # 源代码目录
│   ├── __init__.py             # 包初始化
│   ├── sobel_edge_detection.py # Sobel 边缘检测实现
│   ├── svm_classifier.py       # SVM 分类器
│   ├── cnn_classifier.py       # CNN 分类器（VGG）
│   └── comparison_analysis.py  # 对比分析模块
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖库
└── README.md                   # 项目说明
```

## 🔧 环境要求

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- scikit-learn 1.3+

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 方式一：交互模式（推荐）

```bash
python main.py
```

启动交互式菜单，可选择执行各项操作。

### 方式二：命令行模式

```bash
# Sobel 边缘检测演示
python main.py --mode sobel

# 训练 SVM
python main.py --mode svm

# 训练 CNN
python main.py --mode cnn

# 完整对比分析（自动训练 + 对比）
python main.py --mode compare

# 仅生成对比报告（使用已有结果）
python main.py --mode report

# 检查依赖
python main.py --check-deps
```

### 方式三：直接运行模块

```bash
# 从 src 目录直接运行
python -m src.sobel_edge_detection
python -m src.svm_classifier
python -m src.cnn_classifier
python -m src.comparison_analysis
```

## 📋 功能模块

### 1. Sobel 边缘检测 (`src/sobel_edge_detection.py`)

实现完整的 Sobel 边缘检测算法，包括：
- X/Y 方向梯度计算
- 梯度幅值和方向计算
- 非极大值抑制（NMS）
- 滞后阈值处理
- 特征提取（用于 SVM）

**进度条显示：**
```
处理样本：████████████████████████ 100%
```

### 2. SVM 分类器 (`src/svm_classifier.py`)

使用 Sobel + HOG 特征进行图像分类：
- 自动加载/下载 CIFAR-10 数据集
- 提取 Sobel 边缘特征和 HOG 特征
- 使用 sklearn 的 SVM 进行分类
- 支持多种核函数（linear, rbf, poly）
- **带 tqdm 进度条显示特征提取过程**

**训练进度示例：**
```
提取训练集特征...
  提取特征：████████████████████████ 10000/10000 [02:30<00:00, 66.5 样本/s]
特征提取时间：152.34 秒
特征维度：1234

训练 SVM 模型 (kernel=rbf, C=10.0)...
训练时间：45.67 秒
训练集准确率：0.5234
```

### 3. CNN 分类器 (`src/cnn_classifier.py`)

VGG 风格的卷积神经网络：
- 4 个 VGG 卷积块
- BatchNorm 和 Dropout 正则化
- 数据增强（随机翻转、裁剪、色彩抖动）
- 混合精度训练（AMP）
- 学习率调度（CosineAnnealing）
- **带 tqdm 进度条显示每个 epoch 的训练过程**

**训练进度示例：**
```
Epoch [1/50] - Train Loss: 2.3456, Train Acc: 12.34%, Val Loss: 2.1234, Val Acc: 18.56%, Time: 45.2s
Training: 100%|████████████████████████| 391/391 [00:32<00:00, 12.05it/s, loss=2.345, acc=12.3%]
Validating: 100%|██████████████████████| 79/79 [00:08<00:00, 9.87it/s]
```

### 4. 对比分析 (`src/comparison_analysis.py`)

自动生成对比报告和可视化图表：
- 准确率对比柱状图
- 训练/推理时间对比图
- 综合性能雷达图
- 详细分析报告（Markdown）

## 📊 预期输出

### 准确率对比

| 模型 | 测试准确率 |
|------|-----------|
| SVM + Sobel/HOG | ~55-65% |
| CNN (VGG) | ~85-93% |

### 效率对比

| 指标 | SVM + Sobel/HOG | CNN (VGG) |
|------|-----------------|-----------|
| 训练时间 | ~5-10 分钟 | ~30-60 分钟 |
| 推理时间/样本 | ~50-100ms | ~5-10ms (GPU) |

> 注：实际结果取决于硬件配置和训练参数

## 📈 输出文件

训练完成后，`results/` 目录将包含：

| 文件 | 说明 |
|------|------|
| `sobel_demo.png` | Sobel 边缘检测演示 |
| `svm_confusion_matrix.png` | SVM 混淆矩阵 |
| `cnn_confusion_matrix.png` | CNN 混淆矩阵 |
| `cnn_training_history.png` | CNN 训练曲线 |
| `accuracy_comparison.png` | 准确率对比图 |
| `efficiency_comparison.png` | 效率对比图 |
| `radar_comparison.png` | 雷达对比图 |
| `comparison_report.md` | 详细分析报告 |

`models/` 目录将保存：

| 文件 | 说明 |
|------|------|
| `svm_sobel_model.pkl` | SVM 模型 |
| `cnn_vgg_model.pth` | CNN 模型 |

## 🔬 技术细节

### SVM 特征提取

1. **Sobel 特征**
   - 梯度幅值的统计特征（均值、标准差、最值、分位数）
   - 梯度方向特征
   - 分区域梯度特征（4 象限）

2. **HOG 特征**
   - 使用 OpenCV 的 HOGDescriptor
   - 16x16 block, 8x8 cell, 9 个方向 bin

### CNN 网络架构

```
输入：32x32x3
│
├─ Block 1: Conv(3→64)×2 → MaxPool → Dropout(0.1)    [16x16]
├─ Block 2: Conv(64→128)×2 → MaxPool → Dropout(0.2)  [8x8]
├─ Block 3: Conv(128→256)×3 → MaxPool → Dropout(0.3) [4x4]
├─ Block 4: Conv(256→512)×2 → MaxPool → Dropout(0.3) [2x2]
│
├─ Flatten: 512×2×2 = 2048
├─ FC1: 2048 → 512 → ReLU → Dropout(0.5)
├─ FC2: 512 → 128 → ReLU → Dropout(0.25)
└─ Output: 128 → 10 (softmax)
```

## 📝 CIFAR-10 数据集

| 类别 | 英文 |
|------|------|
| 0 | airplane (飞机) |
| 1 | automobile (汽车) |
| 2 | bird (鸟) |
| 3 | cat (猫) |
| 4 | deer (鹿) |
| 5 | dog (狗) |
| 6 | frog (青蛙) |
| 7 | horse (马) |
| 8 | ship (船) |
| 9 | truck (卡车) |

- 训练集：50,000 张图像（每类 5,000 张）
- 测试集：10,000 张图像（每类 1,000 张）
- 图像尺寸：32×32 彩色

## ⚙️ 参数配置

### SVM 参数

```python
classifier = SVMSobelClassifier(
    kernel='rbf',    # 核函数：linear, rbf, poly, sigmoid
    C=10.0,          # 正则化参数
    gamma='scale'    # RBF 核参数
)
```

### CNN 参数

```python
classifier = CNNClassifier(
    learning_rate=0.001,  # 初始学习率
    weight_decay=1e-4     # L2 正则化
)

classifier.train(
    ...,
    epochs=50,            # 训练轮数
    batch_size=128,       # 批大小
    augment=True          # 数据增强
)
```