from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from itertools import cycle

import csv
data = np.empty((0,23))
target = np.array([])
with open("FallData_Two.csv", "r") as f:

    lines = csv.reader(f)
    for line in lines:
        x = list(map(float, line[:-1]))
        y = int(line[-1])
        data = np.append(data, np.array([x]), axis=0)
        target = np.append(target, np.array([y]))
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)

#hyperparameter --> n_neighbors = 1, 5, 50, 100, 500
neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cosine', metric_params=None, n_jobs=None, n_neighbors=500, p=2, weights='uniform')
neigh.fit(x_train, y_train)

y_pred = neigh.predict(x_test)

matrix = confusion_matrix(y_test, y_pred)
accuracy = (matrix[0,0] + matrix[1,1])/(matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1])
print(matrix)
print("Accuracy: ", accuracy)

target = label_binarize(target, classes=[1, 2, 3])
y_test = label_binarize(y_test, classes=[1, 2, 3])
y_train = label_binarize(y_train, classes=[1, 2, 3])

n_classes = target.shape[1]

classifier = OneVsRestClassifier(neigh)
y_score = classifier.fit(x_train, y_train).predict_proba(x_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for j in range(n_classes - 1):
    fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
    roc_auc[j] = auc(fpr[j], tpr[j])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes - 1)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes - 1):
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
for i, color in zip(range(n_classes - 1), colors):  
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