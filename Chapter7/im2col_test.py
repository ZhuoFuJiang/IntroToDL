import sys, os
sys.path.append(os.pardir)
from common.util import im2col, im2col_v1
import numpy as np


x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
# (9, 75)
print(col1, col1.shape)
col1_v1 = im2col_v1(x1, 5, 5, stride=1, pad=0)
print("====================")
print(col1_v1, col1_v1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
# (90, 75)
print("====================")
print(col2, col2.shape)
print("====================")
col2_v1 = im2col_v1(x2, 5, 5, stride=1, pad=0)
print(col2_v1, col2_v1.shape)
