# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 11:26
# @Author  : Zhuofu Jiang
# @FileName: LogicGates.py
# @Software: PyCharm


import numpy as np


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR_np(x1, x2):
    s1 = NAND_np(x1, x2)
    s2 = OR_np(x1, x2)
    y = AND_np(s1, s2)
    return y


if __name__ == "__main__":
    print(XOR_np(0, 0))
    print(XOR_np(0, 1))
    print(XOR_np(1, 0))
    print(XOR_np(1, 1))
