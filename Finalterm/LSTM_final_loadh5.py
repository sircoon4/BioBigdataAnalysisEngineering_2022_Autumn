# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

imu_data = pd.read_excel("./xlsx/Activity_IMU_2.xlsx", header=None)
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

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

model=tf.keras.models.load_model("lstm_final.h5")
model.summary()

x_test=X[:]; y_test=Y[:]

ev=model.evaluate(x_test, y_test, verbose=0)
print("손실 함수:",ev[0], "MAE:",ev[1])

pred=model.predict(x_test)
#print("평균절댓값백분율오차(MAPE):", sum(abs(y_test-pred)/y_test)/len(x_test))

y_true_fall = y_test[:,3]
y_pred_fall = pred[:,3]

x_range=range(len(y_true_fall))
plt.plot(x_range,y_true_fall[x_range],color='red')
plt.plot(x_range,y_pred_fall[x_range],color='blue')
legend_red = mlines.Line2D([], [], color='red', label='True')
legend_blue = mlines.Line2D([], [], color='blue', label='Predicted')
plt.legend(handles=[legend_red, legend_blue], loc="best")
plt.grid()
plt.show()

from sklearn.metrics import f1_score

y_pred_fall = [2 if d > 1.1 else 1 for d in y_pred_fall]
score = f1_score(y_true_fall, y_pred_fall)
print("f1 score: ", score)