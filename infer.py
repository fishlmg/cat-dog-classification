import cv2
import numpy as np

names = ['Cat', 'Dog']

# Load model
net = cv2.dnn.readNetFromONNX("./model.onnx")
layer_names = net.getUnconnectedOutLayersNames()
print(layer_names)

# Load image
im = cv2.imread("d.png")

# Preprocess
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
im = im.astype(np.float32)
im /= std
blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (224, 224), swapRB=True, crop=False, mean=mean)
# blob = (blob - mean) / std
print(blob.shape)

# Inference
net.setInput(blob)
outs = net.forward(layer_names)
print(outs)

# Postprocess
outs = np.squeeze(outs)
pred = np.argmax(outs)
print(names[pred])



