# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################
import logging
import os
import time

import numpy as np
import wx
import wx.xrc

###########################################################################
## Class MyFrame
###########################################################################
from PIL import Image
from keras.engine.saving import load_model

ImagePath = os.getcwd() + '/image/predict/'
if not os.path.exists(ImagePath):
    os.makedirs(ImagePath)


class MyFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"This is Test Program", pos=wx.DefaultPosition,
                          size=wx.Size(600, 400), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        gSizer1 = wx.GridSizer(0, 2, 0, 0)

        self.m_textCtrl1 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(380, -1), 0)
        gSizer1.Add(self.m_textCtrl1, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        gSizer2 = wx.GridSizer(0, 2, 0, 0)

        self.m_button1 = wx.Button(self, wx.ID_ANY, u"...", wx.DefaultPosition, wx.Size(30, -1), 0)
        gSizer2.Add(self.m_button1, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.m_button2 = wx.Button(self, wx.ID_ANY, u"识别数字", wx.DefaultPosition, wx.DefaultSize, 0)
        gSizer2.Add(self.m_button2, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        gSizer1.Add(gSizer2, 1, wx.ALIGN_RIGHT | wx.EXPAND, 5)

        bSizer1.Add(gSizer1, 1, wx.EXPAND, 5)

        self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(600, 300),
                                           wx.ALIGN_LEFT)
        self.m_staticText1.Wrap(-1)
        bSizer1.Add(self.m_staticText1, 0, wx.ALIGN_TOP | wx.ALL, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.m_button1.Bind(wx.EVT_BUTTON, self.openfile)
        self.m_button2.Bind(wx.EVT_BUTTON, self.predictnumber)

    def __del__(self):
        pass

    # Virtual event handlers, overide them in your derived class
    def openfile(self, event):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.m_textCtrl1.SetValue(dialog.GetPath())
            dialog.Destroy
        event.Skip()

    def predictnumber(self, event):
        path = self.m_textCtrl1.GetValue()
        # print(path)
        logging.info(path)
        [check, name] = checkimage(path)
        # print(check, name)
        if check == 0:
            print('这个文件不存在')
            logging.warning('这个文件不存在')
            self.m_staticText1.SetLabel('这个文件不存在')
        elif check == 2:
            print('不是图片文件，请重试')
            logging.warning('不是图片文件，请重试')
            self.m_staticText1.SetLabel('不是图片文件，请重试')
        else:
            self.m_staticText1.SetLabel(str(predict(ImagePath + name)))
        event.Skip()


def checkimage(imagepath):
    if not os.path.exists(ImagePath):
        os.makedirs(ImagePath)
    size = (28, 28)
    imagename = ''
    # print(imagename)
    if not os.path.exists(imagepath):
        return 0, imagename
    else:
        try:
            img = Image.open(imagepath)
            if img.format != 'PNG':
                imagepath = imagepath.split('.')[0] + '.png'
            if img.size != size:
                img = img.resize(size, Image.ANTIALIAS)
            imagename = os.path.split(imagepath)[1]
            img.save(ImagePath + imagename)
            return 1, imagename
        except Exception as e:
            print(e)
            logging.error(e)
            return 2, imagename


def predict(path=None):
    t = time.asctime(time.localtime(time.time()))
    print(t)
    logging.info(t)
    print(path)
    logging.info(path)
    localpath = os.getcwd()
    modelpath = localpath + '/model/ConvolutionModel.h5'
    if not os.path.exists(modelpath):
        print('模型文件不存在，请检查' + modelpath + '是否存在！')
        logging.warning('模型文件不存在，请检查' + modelpath + '是否存在！')
        return '模型文件不存在，请检查' + modelpath + '是否存在！'
    model = load_model(modelpath)
    im = Image.open(path)
    im.show()
    im = im.convert("L")  # 转灰度
    im = np.array(im)  # 转矩阵
    im = im / 255.  # 归一化
    im1 = np.ones((28, 28))
    im = im1 - im  # 反转
    img = im.reshape(1, - 1, 28, 28)
    predict = model.predict_classes(img)
    print(predict[0])
    logging.info('预测结果为:' + str(predict[0]))
    return predict[0]


def runGUI():
    app = wx.App(False)
    frame = MyFrame(None)
    frame.Show(True)
    # start the applications
    app.MainLoop()


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
fp = logging.FileHandler('./log', encoding='utf-8')
fs = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fp])  # 调用

if __name__ == '__main__':
    runGUI()
