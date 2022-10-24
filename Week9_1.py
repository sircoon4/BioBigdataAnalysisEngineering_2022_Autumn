# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:33:31 2022

@author: sirco
"""

from sklearn.linear_model import Perceptron

X=[[0,0], [0,1], [1,0], [1,1]]
y=[-1,1,1,1]

p=Perceptron()
p.fit(X,y)

print("학습된 퍼셉트론의 매개변수: ", p.coef_, p.intercept_)
print("훈련집합에 대한 예측: ", p.predict(X))
print("정확률 측정: ", p.score(X, y)*100, "%")