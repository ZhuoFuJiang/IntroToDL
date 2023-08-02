# -*- coding: utf-8 -*-
# @Time    : 2023/7/28 13:24
# @Author  : Zhuofu Jiang
# @FileName: show_mnist.py
# @Software: PyCharm

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


# 输出各个数据的形状
print("训练集形状: ", x_train.shape)
print("训练集标签形状: ", t_train.shape)
print("测试集形状: ", x_test.shape)
print("测试集标签形状: ", t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
