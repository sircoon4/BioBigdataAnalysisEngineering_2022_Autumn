import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

df=pd.DataFrame()
filename='./txt/Activity_IMU_1.txt'
df=pd.read_csv(filename,sep=',',header=None)
dfx=df.drop(df.columns[-1],axis=1)
dfy=df.iloc[:,[-1]]
seqx=dfx.to_numpy()
seqy=dfy.to_numpy()

def seq2dataset1(seq,window):
    A=[]
    for i in range(len(seq)-window+1):
        a=seq[i:(i+window)]
        a=a.reshape(1, 2000)
        b=np.zeros([1,1072])
        a=np.concatenate((a,b), axis=1)
        a=a.reshape(32,32,3)
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
#print(X)

Y=seq2dataset2(seqy,w)
#print(Y)

split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

cnn = Sequential()
cnn.add(Conv2D(32,(3,3), activation = 'relu', input_shape = (32,32,3)))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128,activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(1,activation = 'softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer = Adam(), metrics = ['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size = 128, epochs = 12, validation_data = (x_test, y_test), verbose = 2)

res = cnn.evaluate(x_test, y_test, verbose = 0)
print("정확률은" , res[1]*100, "%")