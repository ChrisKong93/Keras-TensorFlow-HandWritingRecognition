# Copyright: Copyright (c) 2019
# Createdon: 2019年5月23日
# Author: ChrisKong
# Version: 1.0
# Title: 一个Python程序

import os
import numpy as np
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.utils import np_utils


def run():
    localpath = os.getcwd()
    (x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')
    model = load_model(localpath + '/model/ConvolutionModel.h5')

    # data pre-processing

    x_test = x_test.reshape(-1, 1, 28, 28) / 255.  # normalize
    # print(x_test.shape)
    # print(x_test[0].reshape(1, - 1, 28, 28).shape)
    img = x_test
    predict = model.predict_classes(img)
    print(predict)
    print(y_test)
    correct_indices = np.nonzero(predict == y_test)[0]
    incorrect_indices = np.nonzero(predict != y_test)[0]

    y_test = np_utils.to_categorical(y_test, num_classes=10)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('test loss', loss)
    print('accuracy', accuracy)

    print("Classified correctly count: {}".format(len(correct_indices)))
    print("Classified incorrectly count: {}".format(len(incorrect_indices)))


if __name__ == '__main__':
    run()
