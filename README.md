# 猫狗图像分类（Cat vs Dog）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

经典的猫狗二分类项目：使用 PyTorch 训练 CNN 图像分类模型，并演示如何用 OpenCV DNN 模块直接加载导出的 ONNX 模型进行推理——无需安装深度学习框架即可部署。

English: Train a cat-vs-dog classifier with PyTorch, export ONNX, and run it through OpenCV DNN. Download the images yourself; this repository does not ship the dataset, weights, or sample outputs.

## ✨ 特性

- **训练**：PyTorch CNN 完成端到端训练（Jupyter Notebook 形式，便于学习）
- **轻量推理**：`infer.py` 使用 `cv2.dnn.readNetFromONNX` 加载模型，仅依赖 OpenCV + NumPy
- **附带讲解**：`explain.md` 说明推理脚本和训练 Notebook，适合入门学习
- 图像归一化（均值/标准差预处理）与输出层解析完整实现

## 🗂 项目结构

```
.
├── cat_dog.ipynb     # 模型训练 Notebook（数据加载 → CNN → 训练 → 导出 ONNX）
├── infer.py          # OpenCV DNN 加载 ONNX 模型进行单图推理
├── explain.md        # infer.py 与 Notebook 的讲解
├── requirements.txt
└── datasets/         # 本地数据，不入库。见下方「数据集」
    └── PetImages/
        ├── Cat/
        └── Dog/
```

## 📂 数据集

仓库不附带图片。`cat_dog.ipynb` 使用 `torchvision.datasets.ImageFolder`，路径是 `datasets/PetImages/`。请自行下载后放成下面的结构，类别名就是文件夹名：

```
datasets/PetImages/
├── Cat/
└── Dog/
```

常见来源有两处，内容都来自 Petfinder 与 Microsoft Research 的 Asirra 猫狗照片，也就是大家说的 Kaggle Dogs vs. Cats。请自己下载，不要把图片提交进本仓库。

1. **[Kaggle Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats)**  
   需要 Kaggle 账号。官方训练集里是同一层目录下的 `cat.0.jpg`、`dog.0.jpg` 这类文件。下载后按文件名前缀分成 `Cat/` 和 `Dog/`，再放到 `datasets/PetImages/`。

2. **[Microsoft：Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)**  
   下载中心上的压缩包名为 `kagglecatsanddogs_5340.zip`（页面标注约 786.7 MB）。解压后就是 `PetImages/Cat` 和 `PetImages/Dog`，把整个 `PetImages` 放进本仓库的 `datasets/` 即可。压缩包里带有数据集自己的许可文件（`CDLA-Permissive-2.0.pdf`），使用图片时以该文件为准。

这个压缩包里有少量无法解码的图片，`ImageFolder` 读到会报错。训练前删掉这些坏文件即可。

`.gitignore` 已忽略 `datasets/`。

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

**方式一：训练模型**

先按上一节放好图片，再打开 Notebook：

```bash
jupyter notebook cat_dog.ipynb
```

按 Notebook 顺序执行：数据预处理 → 构建 CNN → 训练 → 导出 `model.pth` 和 `model.onnx`。这两个文件留在本地。

**方式二：OpenCV 推理（无需 PyTorch）**

`infer.py` 读取 `./model.onnx` 和 `d.png`。先用 Notebook 导出模型，再把 `d.png` 换成你要分类的图片。

```bash
python infer.py
```

脚本里的步骤是：

1. `cv2.dnn.readNetFromONNX()` 加载模型
2. `cv2.dnn.blobFromImage()` 做预处理（缩放到 224×224 并归一化）
3. `net.forward()` 前向推理
4. `argmax` 得到 `Cat` 或 `Dog`

## 🏷 分类标签

```python
names = ["Cat", "Dog"]
```

## 📦 依赖

PyTorch、torchvision、OpenCV（`opencv-python`）、NumPy、Jupyter（只给 Notebook 用）。详见 [requirements.txt](requirements.txt)。`infer.py` 运行时只需要 OpenCV 和 NumPy。

## 📝 说明

训练得到的 `.pth` / `.onnx` 不放进仓库。准确率和效果图以你在本地跑完 Notebook 之后的输出为准，这里不另附数字或截图。

## 🔗 相关项目

- [pytorch-learn](https://github.com/fishlmg/pytorch-learn)：PyTorch 学习笔记
- [opencv-learn](https://github.com/fishlmg/opencv-learn)：OpenCV 基础练习
- [shape-color-detection](https://github.com/fishlmg/shape-color-detection)：传统视觉与 CNN 的形状颜色检测

## 📄 许可证

[MIT License](LICENSE)

## 作者

[@fishlmg](https://github.com/fishlmg)
