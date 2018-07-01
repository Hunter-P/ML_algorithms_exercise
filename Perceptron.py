# -*- coding:utf-8 -*-

import numpy as np


class Perceptron:
    def __init__(self):
        self._w = self_b = None

    def fit(self, x, y, sample_weight=None, lr=0.01, epoch=10**6):
        x, y = np.atleast_2d(x), np.array(y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.array(sample_weight)*len(y)
        # 初始化参数
        self._w = np.zeros(x.shape[1])
        self._b = 0
        for _ in range(epoch):
            y_pred = self.predict(x)
            _err = (y_pred != y)*sample_weight
            # 随机梯度下降
            _indices = np.random.permutation(len(y))
            _idx = _indices[np.argmax(_err[_indices])]
            # 若没有被误分类的样本点则完成了训练
            if y_pred == y:
                return
            # 更新参数
            _delta = lr*y[_idx]*sample_weight[_idx]
            self._w += _delta*x[_idx]
            self._b += _delta

    def predict(self, x, get_raw_results=False):
        rs = np.sum(self._w*x, axis=1) + self._b
        if not get_raw_results:
            return np.sign(x)
        return rs
