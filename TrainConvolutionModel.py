# Copyright: Copyright (c) 2019
# Createdon: 2019年5月23日
# Author: ChrisKong
# Version: 1.0
# Title: 一个Python程序

import os

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Activation, MaxPooling2D, Convolution2D, Flatten, Dense
from keras.utils import np_utils


def buildmodel():
    model = Sequential()
    model.add(Convolution2D(
        batch_input_shape=(32, 1, 28, 28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',  # Padding method
        data_format='channels_first',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',  # Padding method
        data_format='channels_first',
    ))
    model.add(Convolution2D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding='same',  # Padding method
        data_format='channels_first',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',  # Padding method
        data_format='channels_first',
    ))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def train():
    localpath = os.getcwd()
    # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
    (x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')

    # data pre-processing
    # X shape (60,000 28x28), y shape (10,000, )
    x_train = x_train.reshape(-1, 1, 28, 28) / 255.
    x_test = x_test.reshape(-1, 1, 28, 28) / 255.  # normalize
    # print(x_test.shape)
    # print(x_test[1].shape)
    # print(x_test[1].reshape(1, - 1, 28, 28).shape)
    # exit()
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    model = buildmodel()  # 导入模型结构

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss', loss)
    print('accuracy', accuracy)

    # 保存模型
    if not os.path.exists(localpath + '/model'):
        os.mkdir(localpath + '/model')
    model.save('./model/ConvolutionModel.h5')  # HDF5文件，pip install h5py


if __name__ == '__main__':
    train()
