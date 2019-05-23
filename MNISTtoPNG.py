# Copyright: Copyright (c) 2019
# Createdon: 2019年5月17日
# Author: ChrisKong
# Version: 1.0
# Title: 一个Python程序

import os
import time

import numpy as np
from PIL import Image
from keras.datasets import mnist


def get_test_image():
    localpath = os.getcwd()
    filepath = localpath + '/image/test/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    (x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')
    im = np.ones((28, 28)) * 255
    for i in range(len(x_test)):
        img = Image.fromarray(im - x_test[i])
        # 模式
        # 1        1位像素，黑和白，存成8位的像素
        # L        8位像素，黑白
        # P        8位像素，使用调色板映射到任何其他模式
        # RGB      3×8位像素，真彩
        # RGBA     4×8位像素，真彩 + 透明通道
        # CMYK     4×8位像素，颜色隔离
        # YCbCr    3×8位像素，彩色视频格式
        # I        32位整型像素
        # F        32位浮点型像素
        img = img.convert('RGBA')  # F模式转RGBA
        filename = str(y_test[i]) + '_' + str(int(round(time.time() * 1000)))
        print(filename)
        img.save(filepath + filename + '.png')
        # time.sleep(0.1)
    print('Finished')


if __name__ == '__main__':
    get_test_image()
