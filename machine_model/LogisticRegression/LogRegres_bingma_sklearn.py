# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 10:43
# @Author  : gaofeil
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
"""
函数说明:使用Sklearn构建Logistic回归分类器

"""
def colicSklearn():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
	colicSklearn()

