# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imu_data = pd.read_excel("./xlsx/Activity_IMU_1.xlsx", header=None)
seq=imu_data[[0, 1, 2, 20]].to_numpy()

def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=100
h=1

X, Y = seq2dataset(seq,w,h)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import KFold
n_epoch=10
batch_siz=32
k=5

accuracy=[]
for train_index, val_index in KFold(k).split(X):
    xtrain, xval=X[train_index],X[val_index]
    ytrain, yval=Y[train_index],Y[val_index]
    
    model=Sequential()
    model.add(LSTM(units=128, activation='relu', input_shape=X[0].shape))
    model.add(Dense(4))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    model.fit(xtrain, ytrain, epochs=n_epoch, batch_size=batch_siz, validation_data=(xval, yval), verbose=2)
    accuracy.append(model.evaluate(xval,yval,verbose=0)[1])

print(accuracy)
