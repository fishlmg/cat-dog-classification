# 猫狗图像分类（Cat vs Dog）

基于 PyTorch 训练的猫狗二分类模型，并提供 OpenCV DNN 加载 ONNX 模型的推理示例。

## 文件结构

```
├── cat_dog.ipynb   # 模型训练 Notebook（PyTorch CNN）
├── infer.py        # OpenCV dnn 加载 ONNX 模型进行单图推理
└── explain.txt     # infer.py 逐行讲解（学习笔记）
```

## 快速开始

```bash
pip install -r requirements.txt

# 方式一：OpenCV DNN 推理
python infer.py

# 方式二：打开 Notebook 查看 / 重训模型
jupyter notebook cat_dog.ipynb
```

## 依赖

PyTorch、torchvision、OpenCV、NumPy，详见 `requirements.txt`。

## 说明

训练好的模型权重不包含在仓库中（`.gitignore` 已排除）。
