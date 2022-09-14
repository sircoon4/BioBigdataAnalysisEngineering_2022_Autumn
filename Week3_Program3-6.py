# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:35:54 2022

@author: sirco
"""

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit=datasets.load_digits()
s=svm.SVC(gamma=0.01, C=5)
accuracies=cross_val_score(s, digit.data, digit.target, cv=10)

print(accuracies)
print("정확률(평균)=%0.3f, 표준편차=%0.3f"%(accuracies.mean()*100, accuracies.std()))