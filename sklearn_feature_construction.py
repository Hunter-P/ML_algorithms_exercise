# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans

# 新特征的构造
# 1.在原始特征的基础上做数学运算，如平方，开方， log等
# 2.特征组合，对几个特征做线性组合（相加，相减）
# 3.多项式特征，特征之间相乘、相除
# 4.根据聚类生成新特征


# 用户特征
userFeature_file = r"C:\Users\Administrator\Desktop\preliminary_contest_data\userFeature.csv"
with open(userFeature_file, 'r') as fl:
    reader = fl.readlines()
    head = reader[0].strip().split(',')
    data = []
    for row in reader[1:]:
        data.append(row.strip().split(','))
    user_feature = pd.DataFrame(data=data, columns=head)
    # print(user_feature.isnull().any())  # 有缺失值


# 均值填补 user_feature的缺失值
features = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house']
part_user_feature = user_feature[features].replace('', np.nan)
part_user_feature = part_user_feature.fillna(-1)
part_user_feature = pd.DataFrame(part_user_feature, dtype=int).replace(-1, np.nan)
part_user_feature = part_user_feature.fillna(part_user_feature.mean())
print("缺失值填补完毕。")


def newUserFeaturesConstruction(user_feature, part_user_feature):
    """
    构建新的用户特征
    :param user_feature:
    :return:
    """
    newFeatures1 = ['LBS**2', 'age**2', 'carrier**2', 'consumptionAbility**2', 'education**2', 'gender**2', 'house**2']
    newFeatures2 = ['LBS**1/2', 'age**1/2', 'carrier**1/2', 'consumptionAbility**1/2', 'education**1/2', 'gender**1/2',
                    'house**1/2']
    newFeatures3 = ['log/LBS', 'log/age', 'log/carrier', 'log/consumptionAbility', 'log/education', 'log/gender',
                    'log/house']

    for new_f, f in zip(newFeatures1, features):
        try:
            user_feature[new_f] = part_user_feature[f].apply(int)**2
        except:
            raise ValueError
            # print(f)
    print("平方构造新特征完毕！")

    for new_f, f in zip(newFeatures2, features):
        try:
            user_feature[new_f] = part_user_feature[f].apply(int)**0.5
        except:
            print(f)
    print("开方构造新特征完毕！")

    for new_f, f in zip(newFeatures3, features):
        try:
            user_feature[new_f] = part_user_feature[f].apply(int).apply(math.log)
        except:
            print(f)
    print("log新特征完毕！")

    # 特征两两相加
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            try:
                user_feature[features[i]+'+'+features[j]] = part_user_feature[features[i]].apply(int) + \
                                                            part_user_feature[features[j]].apply(int)
            except:
                print(features[i], features[j])
    print("两两相加特征构造完毕！")

    # 特征两两相减
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            try:
                user_feature[features[i]+'-'+features[j]] = part_user_feature[features[i]].apply(int) - \
                                                            part_user_feature[features[j]].apply(int)
            except:
                print(features[i], features[j])
    print("两两相减特征构造完毕！")

    # 特征两两相乘
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            try:
                user_feature[features[i]+'*'+features[j]] = part_user_feature[features[i]].apply(int) * \
                                                            part_user_feature[features[j]].apply(int)
            except:
                print(features[i], features[j])
    print("两两相乘特征构造完毕！")

    # 特征两两相除
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            try:
                user_feature[features[i]+'/'+features[j]] = part_user_feature[features[i]].apply(int) / \
                                                            part_user_feature[features[j]].apply(int)
            except:
                print(features[i], features[j])
    print("两两特征相除构造完毕！")

    # 通过k-means聚类构造新特征，聚类中心数目为1
    clf = KMeans(n_clusters=1)
    clf.fit(part_user_feature)
    center1 = clf.cluster_centers_[0]  # 样本中心
    kmeans_feature = []
    print(center1)

    for i in range(part_user_feature.shape[0]):
        distance1 = np.sqrt(np.sum(np.square(part_user_feature.iloc[i, :] - center1)))
        kmeans_feature.append(distance1)
    user_feature['k-means-1'] = kmeans_feature
    print("聚类特征1类构造完毕！")

    # 通过k-means聚类构造新特征，聚类中心数目为2， 为区别两个簇，用+， -值区分新特征
    clf = KMeans(n_clusters=2)
    clf.fit(part_user_feature)
    center1, center2 = clf.cluster_centers_  # 样本中心
    kmeans_feature = []

    for i in range(part_user_feature.shape[0]):
        distance1 = np.sqrt(np.sum(np.square(part_user_feature.iloc[i, :] - center1)))
        distance2 = np.sqrt(np.sum(np.square(part_user_feature.iloc[i, :] - center2)))
        if distance1 > distance2:
            kmeans_feature.append(-distance2)
        else:
            kmeans_feature.append(distance1)
    user_feature['k-means-2'] = kmeans_feature
    print("聚类特征2类构造完毕！")

    user_feature.to_csv(r"C:\Users\Administrator\Desktop\preliminary_contest_data\userFeature_new1.csv", index=False)

    # 注意，新特征不应对其进行 labelencoder 和 one_hot_encoder，但需要归一化处理。
    # 以上新特征的构造均基于'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house'这7个单数值型特征。
    # 文本特征还需要进一步挖掘。

newUserFeaturesConstruction(user_feature, part_user_feature)


# 广告特征
adFeature_file = r"C:\Users\Administrator\Desktop\preliminary_contest_data\adFeature.csv"
with open(adFeature_file, 'r') as fl:
    reader = fl.readlines()
    head = reader[0].strip().split(',')
    data = []
    for row in reader[1:]:
        data.append([int(i) for i in row.strip().split(',')])
    ad_feature = pd.DataFrame(data=data, columns=head).replace('', np.nan)
    # print(ad_feature.isnull().any())   # 无缺失值


def newAdFeaturesConstruction(ad_feature):
    """
    构建新的广告特征
    :param user_feature:
    :return:
    """
    productId = []
    for i in ad_feature["productId"].apply(int):
        if i == 0:
            productId.append(0)
        else:
            productId.append(1)
    ad_feature["productId"] = productId
    features_name = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productType']

    for f in features_name:
        ad_feature[f+"**2"] = ad_feature[f].apply(int)**2
    print("平方构造新特征完毕！")

    for f in features_name:
        ad_feature[f+"**0.5"] = ad_feature[f].apply(int)**0.5
    print("开方构造新特征完毕！")

    for f in features_name:
        ad_feature['log/'+f] = ad_feature[f].apply(int).apply(math.log)
    print("log构造新特征完毕！")

    for i in range(len(features_name)):
        for j in range(i+1, len(features_name)):
            try:
                ad_feature[features_name[i]+'-'+features_name[j]] = ad_feature[features_name[i]].apply(int) - \
                                                                      ad_feature[features_name[j]].apply(int)
            except:
                print(features_name[i], features_name[j])
    print("两两相减特征构造完毕！")

    for i in range(len(features_name)):
        for j in range(i+1, len(features_name)):
            try:
                ad_feature[features_name[i]+'+'+features_name[j]] = ad_feature[features_name[i]].apply(int) + \
                                                                      ad_feature[features_name[j]].apply(int)
            except:
                print(features_name[i], features_name[j])
    print("两两相加特征构造完毕！")

    for i in range(len(features_name)):
        for j in range(i+1, len(features_name)):
            try:
                ad_feature[features_name[i]+'*'+features_name[j]] = ad_feature[features_name[i]].apply(int) * \
                                                                      ad_feature[features_name[j]].apply(int)
            except:
                print(features_name[i], features_name[j])
    print("两两相乘特征构造完毕！")

    for i in range(len(features_name)):
        for j in range(i+1, len(features_name)):
            try:
                ad_feature[features_name[i]+'/'+features_name[j]] = ad_feature[features_name[i]].apply(int) / \
                                                                      ad_feature[features_name[j]].apply(int)
            except:
                print(features_name[i], features_name[j])
    print("两两相除特征构造完毕！")

    # 通过k-means聚类构造新特征，聚类中心数目为1
    clf = KMeans(n_clusters=1)
    clf.fit(ad_feature[features_name+["productId"]])
    center_p = clf.cluster_centers_[0]  # 样本中心
    kmeans_feature = []
    for i in range(ad_feature.shape[0]):
        distance = np.sqrt(np.sum(np.square(ad_feature[features_name+["productId"]].iloc[i, :] - center_p)))
        kmeans_feature.append(distance)
    ad_feature['k-means-1'] = kmeans_feature
    print("聚类特征1类构造完毕！")

    ad_feature.to_csv(r"C:\Users\Administrator\Desktop\preliminary_contest_data\adFeature_new1.csv", index=False)

newAdFeaturesConstruction(ad_feature)




