# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


"""
tensorflow 实现去噪自编码器, 无监督学习
"""

# xaiver初始化器，fan_in,fan+out分别是输入节点数和输出节点数


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6/(fan_in+fan_out))
    high = constant*np.sqrt(6/(fan_in+fan_out))
    return tf.random_uniform((fan_in+fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussionNoiseAuotencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """
            n_input:输入变量数
            n_hidden：隐含层节点数
            transfer_function：激活函数
            scale：高斯噪声系数,0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐含层输出
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x+scale*tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']))

        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']), self.weights['b2'])

        # 损失函数
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()  # 初始化模型参数
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        初始化权重
        :return:
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype=tf.float32))

        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale})

    def transform(self, X):
        """
        返回隐含层的输出结果
        :param X:
        :return:
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        重构层，将隐含层提取的高阶特征重构为原始数据
        :param hidden:
        :return:
        """
        if not hidden:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        整体运行一遍复原过程，包括了transform和generate两块
        :param X:
        :return:
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale})

    def getWeights(self):
        """
        获取隐含层的权重
        :return:
        """
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """
        获取隐含层的偏置
        :return:
        """
        return self.sess.run(self.weights['b1'])
