# -*- coding: utf-8 -*-
# @Time    : 2023/7/28 13:41
# @Author  : Zhuofu Jiang
# @FileName: function.py
# @Software: PyCharm


import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error_one_hot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def softmax_v1(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1)[:, None]
        y = np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        return y

    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("平方损失", mean_squared_error(np.array(y), np.array(t)))
    print("交叉熵损失", cross_entropy_error(np.array(y), np.array(t)))

    y = [[0.1, 0.05, 0.6, 0.0, 0.05], [0.1, 0.0, 0.1, 0.0, 0.0]]
    print("softmax", softmax(np.array(y)))
    print("softmax_v1", softmax_v1(np.array(y)))

