import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.DataFrame()
filename='Text_Files_version4/txt/Activity_IMU_1.txt'
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import KFold

split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]




k=5


accuracy=[]
for train_index,val_index in KFold(k).split(x_train):
    xtrain,xval=x_train[train_index],x_train[val_index]
    ytrain,yval=y_train[train_index],y_train[val_index]

    model = Sequential()
    model.add(LSTM(units=128,input_shape=(100,20)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    hist=model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_test,y_test),verbose=2)
    accuracy.append(model.evaluate(xval,yval,verbose=0)[1])



    #pred=model.predict(x_test)
    #print("LSTM 평균절댓값백분율오차(MAPE):",sum(abs(y_test-pred)/y_test)/len(x_test))

"""
x_range=range(len(y_test))
plt.plot(x_range, y_test[x_range], color='red')
plt.plot(x_range, pred[x_range], color='red')
plt.grid()
plt.show()
"""

print(accuracy)