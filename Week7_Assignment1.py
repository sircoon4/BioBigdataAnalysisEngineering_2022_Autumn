# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:11:25 2022

@author: sirco
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=700, centers=6,
                       cluster_std=0.75, random_state=0)

print(np.array(X).shape)
print(type(y_true))
#print(X[:, 0])

n_clusters = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in n_clusters]
 
score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]

plt.plot(n_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

print()

np_x = np.array(n_clusters)
np_y = np.array(score)

slope, intercept = np.polyfit(np_x, np_y, 1)

print("slope", slope, "intercept", intercept)
print("slope/10", slope/10)

optimizedNumber = 0
for i in n_clusters[:-2]:
    pointSlope = score[i+1] - score[i]
    print("slope of", i, pointSlope)
    if(pointSlope > slope/5):
        optimizedNumber = i
        break
    
print("optimizedNumber", optimizedNumber)

plt.scatter(X[:, 0], X[:, 1], s=50);

kmeans = KMeans(n_clusters=optimizedNumber)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('K-means Clustering')
plt.show()
