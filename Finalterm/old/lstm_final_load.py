# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.DataFrame()
filename='./txt/Activity_IMU_2.txt'
df=pd.read_csv(filename,sep=',',header=None)
dfx=df.drop(df.columns[-1],axis=1)
dfy=df.iloc[:,[-1]]
seqx=dfx.to_numpy()
seqy=dfy.to_numpy()

def seq2dataset1(seq,window):
    A=[]
    for i in range(len(seq)-window+1):
        a=seq[i:(i+window)]
        A.append(a)
    return np.array(A)

def seq2dataset2(seq,window):
    A=[]
    for i in range(len(seq)-window+1):
        a=seq[i:(i+window)]
        A.append(max(a))
    return np.array(A)

w=100
X=seq2dataset1(seqx,w)
Y=seq2dataset2(seqy,w)

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

model=tf.keras.models.load_model("lstm_final.h5")
model.summary()

x_test=X[:]; y_test=Y[:]

ev=model.evaluate(x_test, y_test, verbose=0)
print("손실 함수:",ev[0], "MAE:",ev[1])

pred=model.predict(x_test)
print(pred)