# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:04:22 2022

@author: sirco
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target

print(np.array(X).shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

neigh1 = KNeighborsClassifier(metric='cosine', n_neighbors=5)

neigh1.fit(x_train, y_train)

y_pred1 = neigh1.predict(x_test)

scores1 = metrics.confusion_matrix(y_test, y_pred1)

print("cosine")
print(scores1)

print()

neigh2 = KNeighborsClassifier(metric='sqeuclidean', n_neighbors=5)

neigh2.fit(x_train, y_train)

y_pred2 = neigh2.predict(x_test)

scores2 = metrics.confusion_matrix(y_test, y_pred2)

print("sqeuclidean")
print(scores2)