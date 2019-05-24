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
    |-- TrainConvolutionModel.py 训练自己的Convolution模型
    |-- TrainDenseModel.py 训练自己的Dense模型
    |-- mnist 文件夹里是下载好的mnist数据
        |-- mnist.npz
```
项目中有两个模型，一个全连接训练出来的，一个CNN训练出来的

