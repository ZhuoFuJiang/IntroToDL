import sys, os

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.functions import sigmoid


# 1000个数据
x = np.random.randn(1000, 100)
# 隐藏层的节点数
node_num = 100
# 隐藏层数
hidden_layer_size = 5
# 激活函数的结果
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 0.01
    # xavier初始化
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    z = np.dot(x, w)
    a = sigmoid(z)
    activations[i] = a

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
