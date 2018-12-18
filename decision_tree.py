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


## 决策树模型
DT = DecisionTreeClassifier()
DT.fit(train_X, train_Y)
test_score_dt = DT.score(test_X, test_Y)
valid_score_dt = DT.score(valid_X, valid_Y)

print('测试集的准确率 in DecisionTreeClassififer：', test_score_dt)
print('验证集的准确率 in DecisionTreeClassififer：', valid_score_dt)

test_prob_dt = DT.predict_proba(test_X)
valid_prob_dt = DT.predict_proba(valid_X)

test_auc_dt = roc_auc_score(test_Y_one_hot, test_prob_dt, average='micro')
valid_auc_dt = roc_auc_score(valid_Y_one_hot, valid_prob_dt, average='micro')

print('测试集的AUC in DecisionTreeClassififer：', test_auc_dt)
print('验证集的AUC in DecisionTreeClassififer：', valid_auc_dt)

plt.figure(1)
### 测试集
fpr, tpr, thresholds = metrics.roc_curve(test_Y_one_hot.ravel(),test_prob_dt.ravel())
auc = metrics.auc(fpr, tpr)

#绘图
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#FPR就是横坐标,TPR就是纵坐标
plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u"test data's ROC&AUC in DecisionTreeClassifier ", fontsize=17)
plt.show()

### 验证集
fpr, tpr, thresholds = metrics.roc_curve(valid_Y_one_hot.ravel(),valid_prob_dt.ravel())
auc = metrics.auc(fpr, tpr)

#绘图
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u"valid data's ROC&AUC in DecisionTreeClassifier ", fontsize=17)
plt.show()