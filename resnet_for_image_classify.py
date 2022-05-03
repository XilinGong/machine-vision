import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions



def predict(model, target_size, top_n, img):
    # 输入：
    # model：分类模型  model = ResNet50(weights='imagenet')
    # target_size：输入图像需要resize的大小 target_size = (224, 224)
    # top_n:输出概率最高的几个类 top_n=1
    # img：待预测图像 img = cv2.imread("1.jpg")
    # 输出：
    # 图像预测结果的元组列表（类、描述、概率）
    #调整数组形状：
    if img.size != target_size:
        target_size = tuple(target_size)
        x = cv2.resize(img,target_size)
    # x = np.transpose(x, (2, 0, 1))#调整数组形状
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #预测
    preds = model.predict(x)
    return decode_predictions(preds, top=top_n)[0]

