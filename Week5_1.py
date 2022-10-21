# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:32:21 2022

@author: sirco
"""

import matplotlib.pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

d=datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(d.data, d.target, train_size=0.8)

reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

plt.plot(y_test, y_pred, 'bo', markersize=2)

x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.title("Kang Seokhun")
plt.show()

