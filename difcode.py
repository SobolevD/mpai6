import math

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imshow, show


def MyDifCode(x, e, r):  # x - source image  e - max error rate  r - predictor number
    size = list(np.shape(x))
    size[1] = size[1] + 1
    size = tuple(size)
    y = np.zeros(size)  # recovered signal
    p = np.zeros(size)  # predicted signal
    f = np.zeros(size)  # difference signal
    q = np.zeros(size)  # quantized difference signal
    for i in range(0, len(x), 1):
        for j in range(0, len(x[0]), 1):
            if r == 1:
                p[i][j] = y[i][j - 1]
            if r == 2:
                p[i][j] = 0.5 * (y[i][j - 1] + y[i - 1][j])
            if r == 3:
                p[i][j] = 0.25 * (y[i][j - 1] + y[i - 1][j] + y[i - 1][j - 1] + y[i - 1][j + 1])
            if r == 4:
                p[i][j] = y[i][j - 1] + y[i - 1][j] - y[i - 1][j - 1]
            f[i][j] = x[i][j] - p[i][j]
            q[i][j] = np.sign(f[i][j]) * math.floor((np.abs(f[i][j]) + e)/(2*e + 1))  #  round to down
            y[i][j] = p[i][j] + q[i][j] * (2*e + 1)
    if e == 0:  # task 6 show difference signal for epsilon = 0
        fig = plt.figure(figsize=(20, 10))
        fig.add_subplot(1, 1, 1)
        plt.title(f"f(dif_img) with predictor {r}")
        imshow(f, cmap='gray')  # , vmin=0, vmax=255  autocontrast
        show()
    return q  # massive quantized differences


def MyDifDecode(q, e, r):  # q -  massive quantized differences  e - max error rate  r - predictor number
    size = np.shape(q)
    y = np.zeros(size)   # recovered signal
    p = np.zeros(size)   # predicted signal
    for i in range(0, len(q), 1):
        for j in range(0, len(q[0]) - 1, 1):
            if r == 1:
                p[i][j] = y[i][j - 1]
            if r == 2:
                p[i][j] = 0.5 * (y[i][j - 1] + y[i - 1][j])
            if r == 3:
                if j + 1 == len(x[0]):
                    y[i - 1][j + 1] = 0
                p[i][j] = 0.25 * (y[i][j - 1] + y[i - 1][j] + y[i - 1][j - 1] + y[i - 1][j + 1])
            if r == 4:
                p[i][j] = y[i][j - 1] + y[i - 1][j] - y[i - 1][j - 1]
            y[i][j] = p[i][j] + q[i][j] * (2 * e + 1)
    return y[:, :-1]  # y - decompressed image  cut last column