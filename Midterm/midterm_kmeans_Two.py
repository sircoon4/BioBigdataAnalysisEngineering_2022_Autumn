# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

csvList = []
rowNum, colNum = 0, 0

with open("./FallData_Two.csv", "r") as inFp :
    csvReader = csv.reader(inFp)
    for row_list in csvReader:
        csvList.append(row_list)
        
    rowNum = len(csvList)
    colNum = len(csvList[0])
    
X = np.array([i[:colNum-1] for i in csvList]).astype(np.float64)
y = np.array([i[colNum-1] for i in csvList]).astype(np.int8)

# init, random_state, algorithm
hyperParameters = [
        ['k-means++', 0, 'full'],
        ['k-means++', 42, 'full'],
        ['k-means++', 0, 'elkan'],
        ['k-means++', 42, 'elkan'],
        ['random', 0, 'full'],
        ['random', 42, 'full'],
        ['random', 0, 'elkan'],
        ['random', 42, 'elkan'],
        ['random', 142, 'elkan'],
        ['random', 242, 'elkan'],
    ]

accuracy = []
for hp in hyperParameters:
    print("init = ", hp[0], ", random_state = ", hp[1], ", algorithm = ", hp[2])
    kmeans = KMeans(n_clusters=2, init=hp[0], random_state=hp[1], algorithm=hp[2])
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    y_predict = y_kmeans + 1
    
    scores = metrics.confusion_matrix(y, y_predict)
    #print(scores)
    
    ableClass = [
        [0, 1],
        [1, 0]
    ]
    
    ableClass = np.array(ableClass)
    
    max = 0
    maxIndex = 0
    index = 0
    for c in ableClass:
        totalPositive = scores[c[0], 0] + scores[c[1], 1]
        #print(totalPositive)
        if totalPositive > max :
            max = totalPositive
            maxIndex = index
        index += 1        
    
    #print(maxIndex)
    
    y_predict = [ableClass[maxIndex, i-1] + 1 for i in y_predict]
    
    print("confusion_matrix")
    scores = metrics.confusion_matrix(y, y_predict)
    print(scores)
    
    accuracyValue = (scores[0,0] + scores[1,1])/scores.sum()
    print("accuracy is", accuracyValue)
    print()
    
    accuracy.append(accuracyValue)

print(accuracy)
plt.bar(np.arange(10), accuracy)
plt.ylim(0.7, 0.71)
plt.xticks(np.arange(10), range(10))
plt.show()
