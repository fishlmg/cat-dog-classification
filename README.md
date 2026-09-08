# 猫狗图像分类（Cat vs Dog）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

经典的猫狗二分类项目：使用 PyTorch 训练 CNN 图像分类模型，并演示如何用 OpenCV DNN 模块直接加载导出的 ONNX 模型进行推理——无需安装深度学习框架即可部署。

## ✨ 特性

- **训练**：PyTorch CNN 完成端到端训练（Jupyter Notebook 形式，便于学习）
- **轻量推理**：`infer.py` 使用 `cv2.dnn.readNetFromONNX` 加载模型，仅依赖 OpenCV + NumPy
- **附带讲解**：`explain.txt` 对推理脚本逐行注释，适合入门学习
- 图像归一化（均值/标准差预处理）与输出层解析完整实现

## 🗂 项目结构

```
.
├── cat_dog.ipynb    # 模型训练 Notebook（数据加载 → CNN → 训练 → 导出 ONNX）
├── infer.py         # OpenCV DNN 加载 ONNX 模型进行单图推理
└── explain.txt      # infer.py 逐行讲解（学习笔记）
```

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

**方式一：训练模型**

```bash
jupyter notebook cat_dog.ipynb
```

按 Notebook 顺序执行：数据预处理 → 构建 CNN → 训练 → 导出 ONNX 模型文件。

**方式二：OpenCV 推理（无需 PyTorch）**

```python
# infer.py 核心流程：
# 1. cv2.dnn.readNetFromONNX()  加载模型
# 2. cv2.dnn.blobFromImage()    图像预处理（归一化）
# 3. net.forward()              前向推理
# 4. argmax → Cat / Dog
python infer.py
```

## 🏷 分类标签

```python
names = ["Cat", "Dog"]
```

## 📦 依赖

PyTorch、torchvision、OpenCV（`opencv-python`）、NumPy —— 详见 [requirements.txt](requirements.txt)

## 📝 说明

训练好的模型权重（`.pth` / `.onnx`）不包含在仓库中（`.gitignore` 已排除），可通过 Notebook 自行训练生成。

## 📄 许可证

[MIT License](LICENSE)
