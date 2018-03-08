# <---python--->

#########################################
"""
k 近邻算法
Input:      newInput: vector to compare to existing dataset (1xN)
            dataSet:  size m data set of known vectors (NxM)
            labels:   data set labels (1xM vector)
            k:        number of neighbors to use for comparison

Output:     the most popular class label
"""
#########################################

# 热身
# from numpy import *
# import operator
#
#
# def createDataSet():
#     group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
#
#
# def kNNClassify(newInput, dataSet, labels, k):
#     numSamples = dataSet.shape[0]
#
#     # step 1: calculate Euclidean distance
#     # tile(A, reps): Construct an array by repeating A reps times
#     # the following copy numSamples rows for dataSet
#     # 计算预测样本与训练集各个样本的距离
#     diff = tile(newInput, (numSamples, 1)) - dataSet
#     squaredDiff = diff**2
#     squareDist = sum(squaredDiff, axis=1)
#     distance = squareDist**0.5
#
#     # step 2: sort the distance
#     # argsort() returns the indices that would sort an array in a ascending order
#     sortedDistIndices = argsort(distance)  # np.argsort()函数返回的是数组值从小到大的索引值
#     classCount = {}
#     for i in range(k):
#         # step 3: choose the min k distance
#         voteLabel = labels[sortedDistIndices[i]]
#
#         # step 4: count the times labels occur
#         # when the key voteLabel is not in dictionary classCount, get()
#         # will return 0
#         classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
#
#     # step 5: the max voted class will return
#     maxCount = 0
#     maxIndex = 0
#     for key, value in classCount.items():
#         if value > maxCount:
#             maxCount = value
#             maxIndex = key
#
#     return maxIndex

# -----------------------------------------------------------------------------------

"""
手写数字识别
"""
from numpy import *
import operator
import time
from os import listdir


def classify(inputPoint, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 用tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = tile(inputPoint, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sum(sqDiffMat, axis=1)
    distances = sqDistances**0.5
    sortedDisIndicies = argsort(distances)  # 按distances中元素进行升序排序后得到的对应下标的列表
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDisIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 按classCount里面类别出现的次数排序，从大到小
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回类次数最多的类别
    return sortedClassCount[0][0]


# 数据准备：文本向量化 32x32 -> 1x1024，数字图像文本向量化，这里将32x32的二进制图像文本矩阵转换成1x1024的向量
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


# 构建训练数据集：利用目录trainingDigits中的文本数据构建训练集向量，以及对应的分类向量
# 从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


# 构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    hwLabels = []
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    trainingFileList = listdir(r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据1\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    path = 'C:\\Users\\Administrator\\Desktop\\机器学习资料汇总\\手写数字识别数据\\数据1\\trainingDigits'
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i, :] = img2vector(path+'\\'+fileNameStr)

    return hwLabels, trainingMat


# 测试集数据测试：通过测试testDigits目录下的样本，来计算算法的准确率。
def handwritingTest():
    hwLabels, trainingMat = trainingDataSet()
    testFileList = listdir(r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据1\testDigits')
    errorCount = 0.0  # 错误数
    mTest = len(testFileList)
    t1 = time.time()
    path = 'C:\\Users\\Administrator\\Desktop\\机器学习资料汇总\\手写数字识别数据\\数据1\\trainingDigits'
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)
        vectorUnderTest = img2vector(path+'\\'+fileNameStr)
        # 调用kNN算法进行测试,分类结果
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k=3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1

    print('\nthe total number of tests is: %d' % mTest)
    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is :%f' % (errorCount/mTest))  # 错误率
    t2 = time.time()
    print('cost time:%.2fmin, %0.4fs' % ((t2-t1)//60, (t2-t1) % 60))

if __name__ == '__main__':
    handwritingTest()

"""
总结：kNN算法的数字识别，将训练集构建成一个多维矩阵，标签类别构建成一维列表，
     将测试样本拓展构建成多维矩阵，计算欧氏距离，按欧氏距离大小排列，取前k个距离最小的测试样本，
     对样本标签计数，数量最多的标签即为输出的预测值。
"""

# --------------------------------------------------------------------------------------------------
'''
    使用python解析二进制文件 
'''
import numpy as np
import struct


def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head


if __name__ == "__main__":
    file1 = 'E:/pythonProjects/dataSets/mnist/train-images.idx3-ubyte'
    file2 = 'E:/pythonProjects/dataSets/mnist/train-labels.idx1-ubyte'

    imgs, data_head = loadImageSet(file1)
    print('data_head:', data_head)
    print(type(imgs))
    print('imgs_array:', imgs)
    print(np.reshape(imgs[1, :], [28, 28]))  # 取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦

    print('----------我是分割线-----------')

    labels, labels_head = loadLabelSet(file2)
    print('labels_head:', labels_head)
    print(type(labels))
    print(labels)







