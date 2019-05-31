# Keras-TensorFlow-HandWritingRecognition

通过Keras训练一个手写数字识别的模型，Keras后端为TensorFlow。

文件结构说明如下

```markdown
|-- Keras-TensorFlow-HandWritingRecognition
    |-- MNISTtoPNG.py 把mnist数据保存成png文件(GUI版本需要用到PNG)
    |-- README.md
    |-- model 文件夹下是训练好的model文件
    |-- TestConvolutionModel.py 测试Convolution模型
    |-- TestConvolutionModelGUI.py 测试Convolution模型(GUI版本)
    |-- TestDenseModel.py 测试Dense模型
    |-- TestDenseModelGUI.py 测试Dense模型(GUI版本)
    |-- TestSimpleRNNModel.py 测试SimpleRNN模型
    |-- TestSimpleRNNModelGUI.py 测试SimpleRNN模型(GUI版本)
    |-- TrainConvolutionModel.py 训练自己的Convolution模型
    |-- TrainDenseModel.py 训练自己的Dense模型
    |-- TrainSimpleRNNModel.py 训练自己的SimpleRNN模型
    |-- mnist 文件夹里是下载好的mnist数据
        |-- mnist.npz
```
项目中有三个模型，一个全连接训练出来的，一个CNN训练出来的，一个RNN训练出来的模型

## MNIST简介

先说一下什么是MNIST，MNIST是一个经典的数据集，很多教程都会对它”下手”, 几乎成为一个“典范”。mnist数据集来自美国国家标准与技术研究所， National Institute of Standards and Technology (NIST)。训练集 (training set) 由来自 250 个不同人手写的数字构成，其中 50% 是高中学生，50% 来自人口普查局 (the Census Bureau) 的工作人员。测试集(test set) 也是同样比例的手写数字数据。

MNIST数据库是由Yann提供的手写数字数据库文件，其官方下载地址<http://yann.lecun.com/exdb/mnist/>

## 数据准备

首先我们先准备数据

一般来说，我们可以直接

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

不过有时候由于某些原因导致数据下载很慢，所以我们提前下载好MNIST的数据，网址如下

https://s3.amazonaws.com/img-datasets/mnist.npz

将下载下来的mnist.npz文件放在mnist文件夹中

然后我们这样导入npz文件中的数据

```python
localpath = os.getcwd() # 获取当前路径
(x_train, y_train), (x_test, y_test) = mnist.load_data(localpath + '/mnist/mnist.npz')
```

然后我们得到四组数据，x_train为训练集图片，y_train为训练集图片的标签，同理，x_test为测试集图片，y_test为测试集图片的标签

这样我们的数据就准备好了

## 全连接模型

首先，我们需要对数据进行预处理

```python
# data pre-processing
x_train = x_train.reshape(x_train.shape[0], -1) / 255.  # normalize
x_test = x_test.reshape(x_test.shape[0], -1) / 255.  # 归一化
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
```

数据预处理完以后我们就可以搭建网络结构了

```python
model = Sequential([Dense(32, input_dim=784),
                    Activation('relu'),
                    Dense(16),
                    Activation('relu'),
                    Dense(10),
                    Activation('softmax')])
```

这是一个三层的全连接网络

模型编译

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

optimizer：优化器，如Adam

loss：计算损失，这里用的是交叉熵损失

metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。

开始训练模型

```python
model.fit(x=x_train, y=y_train, epochs=100, batch_size=128)
```

x：输入数据。如果模型只有一个输入，那么x的类型是numpy
 array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array

y：标签，numpy array

batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch

最后评估模型

```python
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss', loss)
print('accuracy', accuracy)
```

保存模型，模型保存在model文件下

```python
# 保存模型
if not os.path.exists(localpath + '/model'):
    os.mkdir(localpath + '/model') # localpath前面以后有了
    model.save('./model/DenseModel.h5')  # HDF5文件，pip install h5py
```

这样我们的dense模型就训练好了

## CNN模型

同样我们先进行数据预处理

```python
x_train = x_train.reshape(-1, 1, 28, 28) / 255.
x_test = x_test.reshape(-1, 1, 28, 28) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
```

然后就可以搭建模型了

``` python
model = Sequential()
# 第一层
model.add(Convolution2D(
    batch_input_shape=(32, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',  # Padding method
    data_format='channels_first',
))
# 激活层
model.add(Activation('relu'))
# 池化
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',  # Padding method
    data_format='channels_first',
))
# 第二层
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',  # Padding method
    data_format='channels_first',
))
# 激活层
model.add(Activation('relu'))
# 池化
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',  # Padding method
    data_format='channels_first',
))
# Flatten()把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
# 全连接层
model.add(Dense(1024))
model.add(Dense(10))
model.add(Activation('softmax'))
```

模型设置好以后，编译模型

``` python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

训练模型

``` python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

评估模型

```python
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss', loss)
print('accuracy', accuracy)
```

最后保存模型

```python
# 保存模型
if not os.path.exists(localpath + '/model'):
    os.mkdir(localpath + '/model')
model.save('./model/ConvolutionModel.h5')  # HDF5文件，pip install h5py
```

这样我们的CNN模型就训练好了

## RNN模型

数据预处理

```python
# data pre-processing
x_train = x_train.reshape(-1, 28, 28) / 255.
x_test = x_test.reshape(-1, 28, 28) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
```

搭建模型

```python
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
```

训练模型并保存

```python
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
```

RNN模型就训练好了