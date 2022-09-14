# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:44:51 2022

@author: sirco
"""

from sklearn import datasets
from sklearn import svm

d=datasets.load_iris()

s=svm.SVC(gamma=0.1, C=10)
s.fit(d.data, d.target)

new_d=[
       [6.4, 3.2, 6.0, 2.5],
       [7.1, 3.1, 4.7, 1.35],
       [4.3, 2.0, 1.0, 0.1],
       [5.84, 3.05, 3.76, 1.20],
       [7.9, 4.4, 6.9, 2.5]
       ]

res=s.predict(new_d)
print("새로운 2개 샘플의 부류는", res)