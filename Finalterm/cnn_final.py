import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

df=pd.DataFrame()
seqx=np.empty((0,20), float)
seqy=np.empty((0,1), float)

for i in range(100):
    filename='./txt/Activity_IMU_'+str(i+1)+'.txt'
    df=pd.read_csv(filename,sep=',',header=None)
    dfx=df.drop(df.columns[-1],axis=1)
    dfy=df.iloc[:,[-1]]
    seqx1=dfx.to_numpy()
    seqy1=dfy.to_numpy()
    seqx=np.append(seqx, seqx1, axis=0)
    seqy=np.append(seqy, seqy1, axis=0)
    
def seq2dataset1(seq,window):
    A=[]
    for i in range(len(seq)-window+1):
        a=seq[i:(i+window)]
        a=a.reshape(1, 2000)
        b=np.zeros([1,28])
        a=np.concatenate((a,b), axis=1)
        a=a.reshape(26,26,3)
        a=cv2.resize(a,dsize=(32,32),interpolation=cv2.INTER_CUBIC)
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

print("X", X.shape)
print("Y", Y.shape)

split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]
#y_train=tf.keras.utils.to_categorical(y_train,5)
#y_test=tf.keras.utils.to_categorical(y_test,5)

k=5
threshold = 0.5

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

f1_scores=[]
for train_index,val_index in KFold(k).split(x_train):
    xtrain,xval=x_train[train_index],x_train[val_index]
    ytrain,yval=y_train[train_index],y_train[val_index]

    cnn = Sequential()
    cnn.add(Conv2D(32,(3,3), activation = 'relu', input_shape = (32,32,3)))
    cnn.add(Conv2D(32,(3,3),activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(512,activation = 'relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(1,activation = 'sigmoid'))
    cnn.compile(loss='categorical_crossentropy',optimizer = Adam(), metrics = ['accuracy'])
    hist = cnn.fit(x_train, y_train, batch_size = 128, epochs = 10, validation_data = (x_test, y_test), verbose = 2)
    loss, accuracy = cnn.evaluate(xval, yval, verbose=0)
    
    pred=cnn.predict(xval)
    pred_class = np.where(pred > threshold, 1, 0)
    
    print(confusion_matrix(yval, pred_class))
    f1_scores.append(f1_score(yval, pred_class))

import matplotlib.pyplot as plt

x_range=range(len(yval))
plt.plot(x_range, yval[x_range], color='red')
plt.plot(x_range, pred_class[x_range], color='blue')
plt.grid()
plt.show()

print(accuracy)

print("f1_score: ", f1_scores)
print(np.mean(f1_scores))
