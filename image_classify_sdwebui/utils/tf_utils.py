#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   tensorflow_utils.py
@Time    :   2023/06/14 22:01:47
@Desc    :   Enter description of this module
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import os
import shutil
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from tensorflow.keras.preprocessing import image
from utils.utils import check_dir_exists


# 加载并预处理图片
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_result(dest_dir, model, class_names, csv_path, input_shape):
    # "C:\\Users\\yhk\\Desktop\\Workspace\\stablediffusion相关代码\\0524随机男跑图结果"
    result_data = []
    # 使用模型预测结果
    out_dir = f"{dest_dir}_训练分类"
    check_dir_exists(out_dir)
    for root, _, files in os.walk(dest_dir):
        for file in files:
            img_path = os.path.join(root, file)
            print(img_path)
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_array = preprocess_image(img_path, target_size=input_shape)
            # 使用训练好的模型进行预测
            predictions = model.predict(img_array)

            print(predictions)
            break

            # 获取预测结果
            predicted_class = np.argmax(predictions, axis=1)
            predicted_class_name = class_names[predicted_class[0]]
            print(predicted_class_name)

            dir_path = os.path.join(out_dir, predicted_class_name)

            check_dir_exists(dir_path)
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(dir_path, img_name)
            shutil.copyfile(img_path, dest_path)

            result_data.append([dest_path, img_name, predicted_class_name])

    df = pd.DataFrame(result_data, columns=[
                      "path", "img_name", "predict_class"])

    df.to_csv(csv_path, index=False, encoding="utf-8")


# 计算精确率及召回率
def precision_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall
