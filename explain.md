# 代码讲解

## infer.py

`infer.py` 用 OpenCV 加载 ONNX 模型，对单张图片做猫狗分类。

1. **导入库**
   - `cv2`：OpenCV，这里用它的 DNN 模块做推理。
   - `numpy`：数值计算。

2. **类别名**
   - `names = ['Cat', 'Dog']`。下标 0 是猫，1 是狗。
   - 这与 Notebook 里 `ImageFolder` 的类别顺序一致：文件夹名排序后是 `Cat`、`Dog`。

3. **加载模型**
   - `cv2.dnn.readNetFromONNX("./model.onnx")` 读取 Notebook 导出的 ONNX 文件。仓库不附带该文件。

4. **输出层名称**
   - `getUnconnectedOutLayersNames()` 取得网络输出层名称并打印。

5. **读取图片**
   - `cv2.imread("d.png")`。把路径换成你自己的图片。

6. **预处理**
   - 均值 `(0.485, 0.456, 0.406)`，标准差 `(0.229, 0.224, 0.225)`，与 Notebook 里的 ImageNet 归一化相同。
   - 图像转为 `float32` 后除以标准差。
   - `cv2.dnn.blobFromImage` 生成网络输入：缩放系数 `1/255`、尺寸 `224×224`、`swapRB=True`（BGR 转 RGB），并把上述均值传给 `mean`。

7. **推理**
   - `setInput` 送入 blob。
   - `forward` 做前向计算并打印输出。

8. **后处理**
   - `np.squeeze` 把输出压成一维。
   - `np.argmax` 取得分最高的下标，打印 `names[pred]`。

## cat_dog.ipynb

Notebook 用 PyTorch 微调 ResNet-18，做猫狗二分类，并导出 ONNX。图像需要自己准备，目录见 README 的「数据集」。

1. **导入**
   - `torch`
   - `torchvision.models`、`torchvision.datasets`、`torchvision.transforms`
   - `DataLoader`
   - `cv2` 有导入，训练单元里没有用到。

2. **设备与模型**
   - `torch.device("mps")`，使用 Apple MPS。
   - `models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)` 加载 ImageNet 预训练权重。
   - 全连接层换成 2 类输出。

3. **数据**
   - 变换：`ToTensor`、缩放到 `224×224`、按 ImageNet 均值和标准差归一化。
   - 另定义了把类别下标变成二维向量的 `Lambda`，当前 `ImageFolder(...)` 调用没有把它传进去。
   - 路径：`datasets/PetImages/`。
   - `ImageFolder` 用子文件夹名作为类别。本地一次运行的输出是 `['Cat', 'Dog']`。

4. **划分**
   - 验证集比例 `0.2`，用 `random_split` 划分。
   - 训练集和验证集的 `batch_size` 都是 32；训练集 `shuffle=True`。

5. **损失与优化器**
   - `torch.nn.MSELoss()`。
   - `torch.optim.Adam`，学习率 `1e-3`。

6. **训练循环**
   - 训练轮数 `epoch = 10`。
   - `train_loop`：训练模式、前向、反传、更新参数。
   - `test_loop`：评估模式，并在 `torch.no_grad()` 下计算。

7. **保存与导出**
   - `torch.save(resnet.state_dict(), "model.pth")` 保存权重。
   - `torch.onnx.export` 把模型写到 `model.onnx`，供 `infer.py` 加载。
   - `.pth` 和 `.onnx` 已在 `.gitignore` 中。
