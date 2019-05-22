import os
import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.utils import np_utils


def run():
    LocalPath = os.getcwd()
    (x_train, y_train), (x_test, y_test) = mnist.load_data(LocalPath + '/mnist/mnist.npz')
    model = load_model(LocalPath + '/model/SequentialModel.h5')

    # data pre-processing
    # x_train = x_train.reshape(x_train.shape[0], -1) / 255.  # normalize
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.  # normalize
    # y_train = np_utils.to_categorical(y_train, num_classes=10)
    # y_test = np_utils.to_categorical(y_test, num_classes=10)
    print(x_test.shape)
    print(x_test[0].reshape(1, -1).shape)
    # exit()
    img = x_test
    # img = x_test[0].reshape(1, -1)
    # print(img)
    predict = model.predict_classes(img)
    print(predict)
    print(y_test)
    correct_indices = np.nonzero(predict == y_test)[0]
    # print(correct_indices)
    incorrect_indices = np.nonzero(predict != y_test)[0]

    y_test = np_utils.to_categorical(y_test, num_classes=10)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('test loss', loss)
    print('accuracy', accuracy)

    print("Classified correctly count: {}".format(len(correct_indices)))
    print("Classified incorrectly count: {}".format(len(incorrect_indices)))


def predict(path=None):
    print(path)
    LocalPath = os.getcwd()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data(LocalPath + '/mnist/mnist.npz')
    model = load_model(LocalPath + '/model/model.h5')
    im = Image.open(path)
    im.show()
    im2 = im.convert("L")
    im3 = np.array(im2)
    print(im3.shape)
    im4 = np.ones((28, 28))
    img = ((im3 / 255)).reshape(1, -1)
    # img = (im4 - (im3 / 255)).reshape(1, -1)
    # print(img.shap)
    # img = x_test[0].reshape(1, -1)
    # print(img)
    predict = model.predict_classes(img)
    print(predict[0])
    return predict[0]


if __name__ == '__main__':
    run()
    # predict('D:\WorkSpace\Python\Keras-TensorFlow-HandWritingRecognition/image/无标题.png')
