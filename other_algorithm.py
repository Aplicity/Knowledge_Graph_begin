from os import listdir
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,auc
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

for fileName in listdir('./dataSet'):
    # print(fileName)
    pass

with open('./dataSet/entity2id.txt') as fr:
    entity2id = dict()
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        entity2id[lineArr[0]] = int(lineArr[1])

with open('./dataSet/relation2id.txt') as fr:
    relation2id = dict()
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        relation2id[lineArr[0]] = int(lineArr[1])


def txt2id_Mat(fileName):
    with open(fileName) as fr:
        data = []
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            data.append(lineArr)
    m = len(data)
    id_Mat = np.zeros((m, 3))
    for i in range(m):
        for j in range(2):
            id_Mat[i, j] = entity2id[data[i][j]]

        id_Mat[i, 2] = relation2id[data[i][2]]
    return id_Mat


train_Mat = txt2id_Mat('./dataSet/train.txt')
test_Mat = txt2id_Mat('./dataSet/test.txt')
valid_Mat = txt2id_Mat('./dataSet/valid.txt')

train_X , train_Y  = train_Mat[:,:2], train_Mat[:,2]
test_X , test_Y  = test_Mat[:,:2], test_Mat[:,2]
valid_X , valid_Y  = valid_Mat[:,:2], valid_Mat[:,2]

test_Y_one_hot = label_binarize(test_Y, np.arange(len(relation2id.keys())))
valid_Y_one_hot = label_binarize(valid_Y, np.arange(len(relation2id.keys())))


## logistics回归
alpha = np.logspace(-2, 2, 20)  #设置超参数范围
LG = LogisticRegressionCV(Cs = alpha, cv = 3, penalty = 'l2')  #使用L2正则化
LG.fit(train_X, train_Y)

test_prob_lg = LG.predict_proba(test_X)
valid_prob_lg = LG.predict_proba(valid_X)

test_auc_dt = roc_auc_score(test_Y_one_hot, test_prob_lg, average='micro')
valid_auc_dt = roc_auc_score(valid_Y_one_hot, valid_prob_lg, average='micro')

print('测试集的AUC in LogisticRegressionCV：', test_auc_lg)
print('验证集的AUC in LogisticRegressionCV：', valid_auc_lg)


## SVM
clf = SVC()
clf.fit(train_X, train_Y)
clf.score(test_X, test_Y)

test_score_svm = clf.score(test_X, test_Y)
valid_score_svm = clf.score(valid_X, valid_Y)
print('测试集的准确率 in SVM：', test_score_svm)
print('验证集的准确率 in SVM：', valid_score_svm)