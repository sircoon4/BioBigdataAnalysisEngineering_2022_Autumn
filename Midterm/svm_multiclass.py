# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 01:58:22 2022

@author: yujin
"""

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from itertools import cycle



loc = "C:/Users/yujin/Documents/SKKU/바이오빅데이터분석공학/과제/중간대체"

csv_data = []
csv_target = []

i = 0
with open(loc + "/FallData_Three.csv", "r") as inFp:
    for inStr in inFp:
        inStr = inStr.strip()
        row_list=inStr.split(',')
        arr = list(map(float, row_list))
        csv_data.append(arr[:-1])
        csv_target.append(int(arr[-1])-1)
       
        
csv_data = np.array(csv_data)
csv_target = np.array(csv_target)

x_train,x_test,y_train,y_test = train_test_split(csv_data, csv_target, train_size = 0.6, random_state=0)


#변경할 hyperparameter: C ... 2, 4, 6 ,8, 10
s=svm.SVC(kernel="linear", gamma=0.001, C=2, probability=True, random_state=0)
s.fit(x_train, y_train)

y_pred = s.predict(x_test)

#Confusion Matrix...
conf = np.zeros((3,3))

for i in range(len(y_pred)):
    conf[y_pred[i]][y_test[i]] += 1
print(conf)
print()

#Accuracy...
no_correct = 0
for i in range(3):
    no_correct += conf[i][i]
accuracy = no_correct/len(y_pred)
print("테스트 집합에 대한 정확률: ", accuracy*100, "%")



#ROC curve...
csv_target = label_binarize(csv_target, classes=[0, 1, 2])
y_test = label_binarize(y_test, classes=[0, 1, 2])
y_train = label_binarize(y_train, classes=[0, 1, 2])
y_pred = label_binarize(y_pred, classes=[0, 1, 2])
n_classes = csv_target.shape[1]


classifier = OneVsRestClassifier(s)
y_score = classifier.fit(x_train, y_train).decision_function(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for j in range(n_classes):
    fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
    roc_auc[j] = auc(fpr[j], tpr[j])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

lw = 2
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):  
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )
plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
