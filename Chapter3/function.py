# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 14:14
# @Author  : Zhuofu Jiang
# @FileName: function.py
# @Software: PyCharm


import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identify_function(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


if __name__ == "__main__":
    # x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)
    # y = sigmoid(x)
    # y = relu(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)
    # plt.show()

    # x = np.array([-1.0, 1.0, 2.0])
    # print(sigmoid(x))

    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    print(np.sum(y))
