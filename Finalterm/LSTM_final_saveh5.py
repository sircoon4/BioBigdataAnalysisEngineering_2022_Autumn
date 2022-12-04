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
print(X.shape, Y.shape)
print(X[0], Y[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

split = int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

model=Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist=model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test), verbose=2)

model.save("lstm_final.h5")

ev=model.evaluate(x_test, y_test, verbose=0)
print("손실 함수:",ev[0], "MAE:",ev[1])

pred=model.predict(x_test)
#print("평균절댓값백분율오차(MAPE):", sum(abs(y_test-pred)/y_test)/len(x_test))

plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()
