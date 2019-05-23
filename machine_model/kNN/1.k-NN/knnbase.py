# -*- coding: UTF-8 -*-
import numpy as np
import operator

def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN分类器

Parameters:
	trainData - 用于分类的数据(测试集)
	testData - 用于训练的数据(训练集)
	labes - 分类标签
	 k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果
"""
def classify0(trainData, testData, labels, k):
	#shape[0]返回dataSet的行数
	dataSetSize = testData.shape[0]
	#np.tile 将原矩阵横向、纵向地复制 下面表示吧inX复制成datasetsize行 1列
	diffMat = np.tile(trainData, (dataSetSize, 1)) - testData
	#二维特征相减后平方
	sqDiffMat = diffMat**2
	#sum()所有元素相加，sum(0)列相加，sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	#开方，计算出距离
	distances = sqDistances**0.5
	#返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,
		# 如果值不在字典中返回默认值。计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序，最前面的为次数最多的种类
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

if __name__ == '__main__':

	group, labels = createDataSet()
	test = [10,200]
	test_class = classify0(test, group, labels, 3)
	print(test_class)
