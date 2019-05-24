# Copyright: Copyright (c) 2019
# Createdon: 2019年5月23日
# Author: ChrisKong
# Version: 1.0
# Title: 一个Python程序

import os

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation, SimpleRNN
from keras.optimizers import Adam
from keras.utils import np_utils


def buildmodel():
    TIME_STEPS = 28
    INPUT_SIZE = 28
    OUTPUT_SIZE = 10
    CELL_SIZE = 50
    LR = 0.001
    model = Sequential()
    model.add(SimpleRNN(
        # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
        # Otherwise, model.evaluate() will get error.
        batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
        output_dim=CELL_SIZE,
        unroll=True,
    ))
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train():
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    localpath = os.getcwd()
    (x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')

    # data pre-processing
    x_train = x_train.reshape(-1, 28, 28) / 255.
    x_test = x_test.reshape(-1, 28, 28) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    model = buildmodel()
    step = 0
    while True:
        # data shape = (batch_num, steps, inputs/outputs)
        x_batch = x_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
        y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
        cost = model.train_on_batch(x_batch, y_batch)
        BATCH_INDEX += BATCH_SIZE
        BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
        if step % 500 == 0:
            # cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
            cost, accuracy = model.evaluate(x_train, y_train, batch_size=y_test.shape[0], verbose=False)
            print('step ' + str(step) + ',train cost is ', cost, ',train accuracy is ', accuracy)
            if accuracy > 0.95:
                testcost, testaccuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
                print('step ' + str(step) + ',test cost is ', testcost, ',test accuracy is ', testaccuracy)
                model.save('./model/SimpleRNNModel.h5')  # HDF5文件，pip install h5py
                break
        step += 1


if __name__ == '__main__':
    train()
