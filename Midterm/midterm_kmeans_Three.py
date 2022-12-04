# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

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

# init, random_state, algorithm
hyperParameters = [
        ['k-means++', 0, 'full'], #1
        ['k-means++', 42, 'full'], #2
        ['k-means++', 0, 'elkan'], #3
        ['k-means++', 42, 'elkan'], #4
        ['random', 0, 'full'], #5
        ['random', 42, 'full'], #6
        ['random', 0, 'elkan'], #7
        ['random', 42, 'elkan'], #8
        ['random', 142, 'elkan'], #9
        ['random', 242, 'elkan'], #10
    ]

accuracy = []
for hp in hyperParameters:
    print("init = ", hp[0], ", random_state = ", hp[1], ", algorithm = ", hp[2])
    kmeans = KMeans(n_clusters=3, init=hp[0], random_state=hp[1], algorithm=hp[2])
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
    
    print("confusion_matrix")
    scores = metrics.confusion_matrix(y, y_predict)
    print(scores)
    
    accuracyValue = (scores[0,0] + scores[1,1] + scores[2,2])/scores.sum()
    print("accuracy is", accuracyValue)
    print()
    
    accuracy.append(accuracyValue)

print(accuracy)
plt.bar(np.arange(10), accuracy)
plt.ylim(0.43, 0.455)
xlist = [i for i in range(10)]
xlist = np.array(xlist)
plt.xticks(np.arange(10), xlist + 1)
plt.show()
