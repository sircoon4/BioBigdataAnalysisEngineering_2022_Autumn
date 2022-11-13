import openpyxl
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split
import pydotplus
from IPython.display import Image
import graphviz
import csv

import sys
#np.set_printoptions(threshold=sys.maxsize)

data = np.empty((0,23))
target = np.array([])
with open("./FallData_Three.csv", "r") as f:
    
    lines = csv.reader(f)
    for line in lines:
        x = list(map(float, line[:-1]))
        y = int(line[-1])
        data = np.append(data, np.array([x]), axis=0)
        target = np.append(target, np.array([y]))
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state=0)

# max_features=2, 6, 10, 16, 23
dt = DecisionTreeClassifier(criterion='entropy',max_features=23, random_state=0)
dt = dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

#print("Desirable Result = \n", y_test)
#print("Prediction Result = \n", dt_pred)

print(confusion_matrix(y_test, dt_pred))

no_correct = 0
for i in range (3):
    no_correct += confusion_matrix(y_test, dt_pred)[i][i]
    
ac = no_correct/len(dt_pred)
print("Accuracy : ",ac)

import matplotlib.pyplot as plt

#plt.figure()
#plot_tree(dt,impurity=True, filled=True, rounded=True)
#plt.show()

from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

target = label_binarize(target, classes=[1, 2, 3])
y_test = label_binarize(y_test, classes=[1, 2, 3])
y_train = label_binarize(y_train, classes=[1, 2, 3])

n_classes = target.shape[1]

classifier = OneVsRestClassifier(dt)
y_score = classifier.fit(x_train, y_train).predict_proba(x_test)
#print("y_score", y_score)

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
colors = cycle(["red", "green", "blue"])
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
