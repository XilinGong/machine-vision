import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


def predict(model, target_size, top_n, img):
    if img.size != target_size:
        x = cv2.resize(img, target_size)
    # x = np.transpose(x, (2, 0, 1))#调整数组形状
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # 预测
    preds = model.predict(x)
    return decode_predictions(preds, top=top_n)[0]

