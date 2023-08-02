# -*- coding: utf-8 -*-
# @Time    : 2023/7/28 14:04
# @Author  : Zhuofu Jiang
# @FileName: gradient.py
# @Software: PyCharm


import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x-h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x+h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)

    # print("{}处的梯度: ".format(5), numerical_diff(function_1, 5))
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # plt.plot(x, y)
    # plt.show()

    print("多元参数梯度: ", numerical_gradient(function_2, np.array([3.0, 4.0])))

    init_x = np.array([-3.0, 4.0])
    print("梯度下降法: ", gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))