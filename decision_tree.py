# ---python---
# -*- coding:utf-8 -*-

"""
函数 TreeGenerate(D,A)
生成节点node:
if D 中的样本全属于同一类别 C then
    将node标记为C类叶子节点；return
end if
if A = ϕ OR D 中样本在 A 上取值相同 then
    将 node 标记为叶节点，其类别标记为 D 中样本数最多的类；return
end if
从A中选则最优划分属性a∗;
for a* 的每一个值a*(v) do
    为node生成一个分支；令Dv表示D中在a*上取值为a*(v)的样本子集；
    if Dv 为空 then
        将分支节点标记为叶节点，其类别标记为D中样本最多的类；return
    else
        以Tree(Dv,A\{a*})为分支节点
    end if
end for
"""
from math import log
import operator
import csv
import numpy as np
from collections import defaultdict
import heapq
from operator import itemgetter


class DT(object):
    def __init__(self, practice_set, test_set):
        self.pset_file = practice_set
        self.tset_file = test_set
        self.pdata = []
        self.tdata = []
        self.labels = []

    # 导入数据，生成训练集数组
    # outlook->0:sunny|1:overcast|2:rain
    # windy->0:f|1:t
    # play golf->y:1|n:0
    def create_dataset(self):
        with open(self.pset_file, 'r') as pf:
            reader = csv.reader(pf)
            header = next(reader)
            for row in reader:
                row[1] = int(row[1])
                row[2] = int(row[2])
                if row[0] == 'sunny':
                    row[0] = 0
                elif row[0] == 'overcast':
                    row[0] = 1
                else:
                    row[0] = 2
                if row[3] == 'f':
                    row[3] = 0
                else:
                    row[3] = 1
                if row[4] == 'y':
                    row[4] = 1
                else:
                    row[4] = 0

                self.pdata.append(row)
        self.pdata_array = np.asarray(np.array(self.pdata), dtype=int)
        return self.pdata_array

    # 导入数据，生成测试集数组
    def create_test_dataset(self):
        with open(self.tset_file, 'r') as pf:
            reader = csv.reader(pf)
            header = next(reader)
            for row in reader:
                row[1] = int(row[1])
                row[2] = int(row[2])
                if row[0] == 'sunny':
                    row[0] = 0
                elif row[0] == 'overcast':
                    row[0] = 1
                else:
                    row[0] = 2
                if row[3] == 'f':
                    row[3] = 0
                else:
                    row[3] = 1
                if row[4] == 'y':
                    row[4] = 1
                else:
                    row[4] = 0

                self.tdata.append(row)
        self.tdata_array = np.asarray(np.array(self.tdata), dtype=int)
        return self.tdata_array

    # 计算训练集每个属性的信息增益,并返回最大信息增益的属性和划分值
    # dataset：np数组
    def choose_best_feature_split(self, dataset):
        # 计算当前样本集的信息熵
        length = len(dataset[:, 4])  # 当前样本总数量
        num_y = sum(dataset[:, 4])  # play golf 类别为1（y）的数量
        num_n = length - num_y  # play golf 类别为0（n）的数量
        if num_y != 0 and num_n != 0:
            ent = -(num_y/length*log(num_y/length, 2) +
                    num_n/length*log(num_n/length, 2))
        else:
            ent = 0

        condition_ent_list = []  # 各个属性的熵[[attri, divide, c_ent], []...]

        for i in range(len(dataset[0])-1):  # 计算各个属性的条件熵
            attri = dataset[:, i]
            if i == 0 or i == 3:
                d, c = self.ent_ls(i, attri, dataset)
                condition_ent_list.append([i, d, c])

            else:
                d, c = self.ent_lx(i, attri, dataset)
                condition_ent_list.append([i, d, c])
        need_attri = 0
        need_condition_ent = 1
        need_divide = 0
        for j in condition_ent_list:
            if j[2] < need_condition_ent:
                need_condition_ent = j[2]
                need_attri = j[0]
                need_divide = j[1]
        gain = ent - need_condition_ent
        return need_attri, need_divide

    # 计算离散属性的条件熵,返回None（表示无划分值）和条件熵
    # 描述：将训练集按属性划分，导入字典数据结构，属性的取值用集合表示
    # attri：[0,1,...],n表示第n个属性
    # data:{:[[],[]...], ...},划分后的样本, dataset:np数组
    def ent_ls(self, n, attri, dataset):
        length = len(dataset)  # 当前训练集样本的数量
        condition_ent = 0  # 条件熵
        data = defaultdict(list)
        attri_set = set()  # 属性的取值集合

        for i in attri:
            attri_set.add(i)

        for j in attri_set:
            num_y = 0
            for k in dataset:
                if k[n] == j:
                    data[j].append(k)  # data:{:[[],[]...], ...},划分后的样本
                    if k[4] == 1:
                        num_y += 1  # play golf 类别为1（y）的数量

            j_length = len(data[j])  # 属性取某值时样本的数量
            num_n = j_length - num_y  # play golf 类别为0（n）的数量
            if num_y != 0 and num_n != 0:
                condition_ent += -(j_length/length*(num_y/j_length*log(num_y/j_length, 2) +
                                                    num_n/j_length*log(num_n/j_length, 2)))
            else:
                condition_ent += 0

        return None, condition_ent

    # 计算连续属性的条件熵，返回划分值和条件熵（条件熵最小）
    # 描述：计算每个区间的属性划分值，c=(a1+a2)/2,将训练集按属性划分，导入字典数据结构
    # attri：[a1,a2,...],n表示第n个属性
    # data:{:[[],[]...], ...},划分后的样本, dataset:np数组
    def ent_lx(self, n, attri, dataset):
        divide = set()  # 划分值的集合
        attri.sort()  # 对属性值进行排序
        condition_ent_dict = {}  # 各个划分值下的条件熵{a1:ent, a2:...}
        length = len(dataset)  # 当前训练集样本的数量

        for i in range(len(attri)-1):
            divide.add((attri[i]+attri[i+1])/2)

        for j in divide:
            num_y1 = 0
            num_y2 = 0
            data = defaultdict(list)
            condition_ent = 0
            for k in dataset:
                if k[n] <= j:
                    data['less'].append(k)
                    if k[4] == 1:
                        num_y1 += 1  # play golf 类别为1（y）的数量,小于划分值的样本
                else:
                    data['more'].append(k)
                    if k[4] == 1:
                        num_y2 += 1  # play golf 类别为1（y）的数量，大于划分值的样本

            j_length1 = len(data['less'])  # 属性小于某划分值时样本的数量
            j_length2 = len(data['more'])  # 属性大于某划分值时样本的数量
            num_n1 = j_length1 - num_y1  # play golf 类别为0（n）的数量,小于划分值的样本
            num_n2 = j_length2 - num_y2  # play golf 类别为0（n）的数量,大于划分值的样本

            if 0 not in [num_n1, num_n2, num_y1, num_y2]:
                condition_ent = -(j_length1/length*(num_y1/j_length1*log(num_y1/j_length1, 2) +
                                                  num_n1/j_length1*log(num_n1/j_length1, 2))) - \
                                (j_length2/length*(num_y2/j_length2*log(num_y2/j_length2, 2) +
                                                  num_n2/j_length2*log(num_n2/j_length2, 2)))
            elif num_n1 == 0 or num_y1 == 0:
                if num_n2 != 0 and num_y2 != 0:
                    condition_ent = -(j_length2/length*(num_y2/j_length2*log(num_y2/j_length2, 2) +
                                                      num_n2/j_length2*log(num_n2/j_length2, 2)))
                else:
                    condition_ent = 0
            else:
                if num_n2 == 0 or num_y2 == 0:
                    condition_ent = -(j_length1/length*(num_y1/j_length1*log(num_y1/j_length1, 2)))
            condition_ent_dict[j] = condition_ent

        # print(heapq.nsmallest(1, condition_ent_dict.items(), key=itemgetter(1)))
        return heapq.nsmallest(1, condition_ent_dict.items(), key=itemgetter(1))[0]

    # 构造决策树
    # dataset：np数组
    def treeGenerate(self, dataset):
        class_list = [x[-1] for x in dataset]
        # D中样本类别完全相同，停止划分
        if class_list.count(class_list[0]) == len(class_list):
            print('class:', class_list[0], dataset)
            return class_list[0]

        no_attri_next, divide_next = self.choose_best_feature_split(dataset)  # 选择划分属性
        data = self.divide_dataset(no_attri_next, divide_next, dataset)  # 划分数据集 {：[array([]),...], :...}

        # D中样本在 A 上取值相同，停止划分
        for i in data:
            if len(data[i]) == len(dataset):
                class_list = [x[-1] for x in data[i]]

                if class_list.count(class_list[0]) > len(class_list)-class_list.count(class_list[0]):
                    print('class:', class_list[0], dataset)
                    return class_list[0]
                else:
                    if class_list[0] == 0:
                        print('class:', 1, dataset)
                        return 1
                    else:
                        print('class:', 0, dataset)
                        return 0

        self.labels.append([no_attri_next, divide_next])
        dataset = self.convert_format(data)  # 转换成数组格式

        for i in dataset:
            self.treeGenerate(i)

    # 根据属性将数据集划分
    # data:{:[[],[]...], ...} 划分后的数据集
    # no_attri表示第几个属性
    def divide_dataset(self, no_attri, divide, dataset):
        length = len(dataset)  # 当前训练集样本的数量
        data = defaultdict(list)
        attri_set = set()  # 属性的取值集合

        if divide == None:
            for i in dataset[:, no_attri]:
                attri_set.add(i)

            for j in attri_set:
                for k in dataset:
                    if k[no_attri] == j:
                        data[j].append(k)

        else:
            for k in dataset:
                if k[no_attri] <= divide:
                    data['less'].append(k)
                else:
                    data['more'].append(k)

        return data

    # 对data：{:[[],[]...], ...}格式转化为多维数组,每个类别对于一个多维数组[array(),array()...]
    def convert_format(self, data):
        return [np.vstack(x) for x in data.values()]


def main():
    practice_set = r'C:\Users\Administrator\Desktop\practice_set.csv'
    test_set = r'C:\Users\Administrator\Desktop\test_set.csv'

    dt = DT(practice_set, test_set)
    dt.create_dataset()
    dt.create_test_dataset()
    print(dt.pdata_array)
    dt.treeGenerate(dt.pdata_array)
    print("labels:", dt.labels)

main()
















