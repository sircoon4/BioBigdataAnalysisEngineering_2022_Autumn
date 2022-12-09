import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
X = np.array([]); Y= np.array([])

for i in range(1, 30):
    df=pd.DataFrame()
    filename='./txt/Activity_IMU_' + str(i) + '.txt'
    df=pd.read_csv(filename,sep=',',header=None)
    dfx=df.drop(df.columns[-1],axis=1)
    dfy=df.iloc[:,[-1]]
    seqx=dfx[[1, 2, 3, 12, 13, 14]].to_numpy()
    seqy=dfy.to_numpy()
    
    if i == 1 :
        X = seq2dataset1(seqx,w)
        Y = seq2dataset2(seqy,w) - 1
    else :
        X = np.append(X, seq2dataset1(seqx,w), axis=0)
        Y = np.append(Y, seq2dataset2(seqy,w) - 1, axis=0)

print("X", X.shape)
print("Y", Y.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import KFold

split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

k=5

f1_scores=[]
for train_index,val_index in KFold(k).split(x_train):
    xtrain,xval=x_train[train_index],x_train[val_index]
    ytrain,yval=y_train[train_index],y_train[val_index]
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(100, 6)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['acc', Precision(name="precision"), Recall(name="recall")])
    hist=model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test,y_test),verbose=2)
    loss, accuracy, precision, recall = model.evaluate(xval, yval, verbose=0)

    f1_score = 2 * (precision * recall) / (precision + recall)
    f1_scores.append(f1_score)


print("f1_score: ", f1_scores)




