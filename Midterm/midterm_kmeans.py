# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn import svm, datasets

csvList = []
rowNum, colNum = 0, 0

with open("./FallData_Three.csv", "r") as inFp :
    csvReader = csv.reader(inFp)
    for row_list in csvReader:
        csvList.append(row_list)
        
    rowNum = len(csvList)
    colNum = len(csvList[0])
    
X = np.array([i[:colNum-1] for i in csvList]).astype(np.float64)
y = np.array([i[colNum-1] for i in csvList]).astype(np.int8)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

y_predict = y_kmeans + 1

scores = metrics.confusion_matrix(y, y_predict)
#print(scores)

ableClass = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0]
]

ableClass = np.array(ableClass)

max = 0
maxIndex = 0
index = 0
for c in ableClass:
    totalPositive = scores[c[0], 0] + scores[c[1], 1] + scores[c[2], 2]
    #print(totalPositive)
    if totalPositive > max :
        max = totalPositive
        maxIndex = index
    index += 1        

#print(maxIndex)

y_predict = [ableClass[maxIndex, i-1] + 1 for i in y_predict]

scores = metrics.confusion_matrix(y, y_predict)
print(scores)


