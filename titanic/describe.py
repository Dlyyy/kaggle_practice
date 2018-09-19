#!/usr/bin/python
# coding: utf-8


#特征详情
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root_path = 'E:/PythonProject01/kaggle_practice/titanic/input/'

train_data = pd.read_csv('%s%s' % (root_path, 'train.csv'))
test_data = pd.read_csv('%s%s' % (root_path, 'test.csv'))


print(train_data.head(5),'\n\n\n')
print(train_data.info(),'\n\n\n')
print(test_data.info(),'\n\n\n')
# # 返回数值型变量的统计量
print(train_data.describe())

#特征分析（统计学与绘图）
# 存活人数
print(train_data['Survived'].value_counts())

# 1)数值型数据协方差,corr()函数
# 来个总览,快速了解个数据的相关性
# 相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
train_corr = train_data.drop('PassengerId',axis=1).corr()
print(train_corr)
# 画出相关性热力图
a = plt.subplots(figsize=(15,9))#调整画布大小
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图
plt.show()