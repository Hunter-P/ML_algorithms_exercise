# <---python--->
# -*- coding: utf-8 -*-
"""
朴素贝叶斯分类
"""
# ---------------------------------------------------------------

import numpy as np
import struct


class LoadData(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    # 载入数据集
    def loadImageSet(self):
        binfile = open(self.file1, 'rb')  # 读取二进制文件
        buffers = binfile.read()  # 缓冲
        head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组
        offset = struct.calcsize('>IIII')  # 定位到data开始的位置

        imgNum = head[1]  # 图像个数
        width = head[2]  # 行数，28行
        height = head[3]  # 列数，28

        bits = imgNum*width*height  # data一共有60000*28*28个像素值
        bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
        imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

        binfile.close()
        imgs = np.reshape(imgs, [imgNum, width*height])

        return imgs, head

    # 载入数据集标签
    def loadLabelSet(self):
        binfile = open(self.file2, 'rb')  # 读取二进制文件
        buffers = binfile.read()  # 缓冲
        head = struct.unpack_from('>II', buffers, 0)  # 取前2个整数，返回一个元组
        offset = struct.calcsize('>II')  # 定位到label开始的位置

        labelNum = head[1]  # label个数

        numString = '>' + str(labelNum) + 'B'
        labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

        binfile.close()
        labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

        return labels, head

    # 将标签拓展为10维向量
    def expand_lables(self):
        labels, head = self.loadLabelSet()
        expand_lables = []
        for label in labels:
            zero_vector = np.zeros((1, 10))
            zero_vector[0, label] = 1
            expand_lables.append(zero_vector)
        return expand_lables

    # 将样本与标签组合成数组[[array(data), array(label)], []...]
    def loadData(self):
        imags, head = self.loadImageSet()
        expand_lables = self.expand_lables()
        data = []
        for i in range(imags.shape[0]):
            imags[i] = imags[i].reshape((1, 784))
            data.append([imags[i], expand_lables[i]])
        return data