# -*- coding: UTF-8 -*-
"""
    @Author : Frank.Ren
    @Project : 2022-Machine-Learning-Specialization 
    @Product : PyCharm
    @createTime : 2023/6/18 14:09 
    @Email : sc19lr@leeds.ac.uk
    @github : https://github.com/frankRenlf
    @Description : 
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# 假设训练集目标变量是一个一维数组
y_train = np.array([[1, 2, 3, 4, 5],
                   [5, 7, 8, 9, 10]])

scalerItem = StandardScaler()
scalerItem.fit(y_train)
item_train = scalerItem.transform(y_train)
print(item_train)

x_np = y_train
mean = np.mean(x_np, axis=0)
std = np.std(x_np, axis=0)
print('矩阵初值为：{}'.format(x_np))
print('该矩阵的均值为：{}\n 该矩阵的标准差为：{}'.format(mean,std))
another_trans_data = x_np - mean
another_trans_data = another_trans_data / std
print('标准差标准化的矩阵为：{}'.format(another_trans_data))
print('-----')

y_train = np.array([[1],[2],[3],[4]])
scaler = MinMaxScaler((-1, 1))
scaler.fit(y_train.reshape(-1, 1))
ynorm_train = scaler.transform(y_train.reshape(-1, 1))
print(ynorm_train)
