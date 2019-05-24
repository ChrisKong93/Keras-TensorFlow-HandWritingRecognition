# Copyright: Copyright (c) 2019
# Createdon: 2019年5月17日
# Author: ChrisKong
# Version: 1.0
# Title: 一个Python程序

import os

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.utils import np_utils


def train():
    localpath = os.getcwd()
    (x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')

    # data pre-processing
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.  # normalize
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    model = Sequential([Dense(32, input_dim=784),
                        Activation('relu'),
                        Dense(16),
                        Activation('relu'),
                        Dense(10),
                        Activation('softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=100, batch_size=128)

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss', loss)
    print('accuracy', accuracy)

    # 保存模型
    if not os.path.exists(localpath + '/model'):
        os.mkdir(localpath + '/model')
    model.save('./model/DenseModel.h5')  # HDF5文件，pip install h5py


if __name__ == '__main__':
    train()
