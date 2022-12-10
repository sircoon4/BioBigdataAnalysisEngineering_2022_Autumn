import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

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
X = np.array([]); Y= np.array([])

for i in range(1, 100):
    df=pd.DataFrame()
    filename='./txt/Activity_IMU_' + str(i) + '.txt'
    df=pd.read_csv(filename,sep=',',header=None)
    dfx=df.drop(df.columns[-1],axis=1)
    dfy=df.iloc[:,[-1]]
    seqx=dfx.to_numpy()
    seqy=dfy.to_numpy()
    if np.count_nonzero(seqy == 2) == 0 :
        continue
    
    if i == 1 :
        X = seq2dataset1(seqx,w)
        Y = seq2dataset2(seqy,w) - 1
    else :
        X = np.append(X, seq2dataset1(seqx,w), axis=0)
        Y = np.append(Y, seq2dataset2(seqy,w) - 1, axis=0)

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
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

accuracies=[]
f1_scores=[]
f1_max=0
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
    cnn.compile(loss='binary_crossentropy',optimizer = Adam(), metrics = ['accuracy'])
    hist = cnn.fit(xtrain, ytrain, batch_size = 128, epochs = 10, validation_data = (xval, yval), verbose = 2)
    accuracies.append(cnn.evaluate(xval, yval, verbose = 0)[1])
    
    pred=cnn.predict(xval)
    pred_class = np.where(pred > threshold, 1, 0)
    
    f1Score = f1_score(yval, pred_class)
    f1_scores.append(f1Score)
    
    if f1Score > f1_max :
        f1Score = f1_max
        cnn.save("cnn_final.h5")

print()
print("accuracy:", accuracies)
print("accuracy_mean:", np.mean(accuracies))
print()
print("f1_score:", f1_scores)
print("f1_score_mean:", np.mean(f1_scores))
print()
model=load_model("cnn_final.h5")
model.summary()
print()
pred = model.predict(x_test)
pred_class = np.where(pred > threshold, 1, 0)

cm = confusion_matrix(y_test, pred_class)
print("confusion_matrix")
print(cm)
print()
TP = cm[0][0]; FN = cm[0][1]; FP = cm[1][0]; TN = cm[1][1]
sensitivity = TP/(TP + FN)
print("Sensitivity:", sensitivity)
specificity = TN/(FP + TN)
print("Specificity:", specificity)
print()
fpr, tpr, _ = roc_curve(y_test, pred)
auc = roc_auc_score(y_test, pred)
print("AUC:",auc)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
