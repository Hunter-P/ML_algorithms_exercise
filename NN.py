# <---python--->
# -*- coding: utf-8 -*-
#########################################
"""
手写数字识别, BP神经网络算法
"""
# -------------------------------------------
'''
使用python解析二进制文件
'''
import numpy as np
import struct
import random


class LoadData(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    # 载入训练集
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

    # 载入训练集标签
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


class Network(object):
    def __init__(self, sizes):  # sizes为各层神经元的数量，第一层（输入层），第二层（隐含层），第三层（输出层）
        self.num_layers = len(sizes)  # 层数
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 随机生成高斯分布的偏置
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # 随机生成高斯分布的权值

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        随机梯度下降算法，eta为学习率,epochs为迭代次数
        """
        n_test = 0
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)  # 随机排序
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        运用反向传播算法更新神经网络的权重和偏置，经过一小批次样本更新一次w，b
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in
                        zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in
                        zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        反向传播算法, 返回w， b的梯度
        当样本属于某一类（某个数字）的时候，则该类（该数字）对应的节点为1，而剩下9个节点为0，
        如[0,0,0,1,0,0,0,0,0,0]。
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x.reshape((784, 1))
        activations = [activation]  # 存放激活函数的输出值
        zs = []  # 存放z向量

        for b, w in zip(self.biases, self.weights):  # 求激活函数输出的值activation，第一个为隐含层，第二个为输出层
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y.transpose())*\
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 均方误差对权值求导

        for l in range(2, self.num_layers):  # 隐含层
            z = zs[-l]  # 不是-1
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[0].transpose())  # transpose()转置
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        '''
        计算预测准确的样本个数
        '''
        test_results = [(np.argmax(self.feedforward(x)), y)  # 返回 x 中最大值1的索引
                        for (x, y) in test_data]

        return sum(int(y[0, x] == 1) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    # 当输⼊ z 是⼀个向量或者 Numpy 数组时，Numpy ⾃动地按元素应⽤ sigmoid 函数，即以向量形式
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # sigmoid函数求导
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, a):
        '''
        反馈，向神经元输入a,经过激活函数处理，输出
        '''
        a = a.reshape((784, 1))
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

if __name__ == "__main__":
    file1 = r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据2\train-images.idx3-ubyte'
    file2 = r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据2\train-labels.idx1-ubyte'
    trainingData = LoadData(file1, file2)
    training_data = trainingData.loadData()
    file3 = r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据2\t10k-images.idx3-ubyte'
    file4 = r'C:\Users\Administrator\Desktop\机器学习资料汇总\手写数字识别数据\数据2\t10k-labels.idx1-ubyte'
    testData = LoadData(file3, file4)
    test_data = testData.loadData()
    net = Network([784, 40, 10])
    net.SGD(training_data, 200, 10, 0.06, test_data=test_data)

    print('----------我是分割线-----------')

# -----------------------------------------------------------------------------------------------------
"""
总结： w，b的初始化用随机生成，并服从高斯分布，运用了反向传播算法，通过链式法则，求出损失函数对w，b的偏导（隐含层和输出层），
然后通过小批量梯度下降法，不断迭代替换w，b。

      小批量梯度下降：将样本分为x批，求每一批样本的梯度后，再对w,b 进行一次更新。
"""















