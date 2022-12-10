import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

df=pd.DataFrame()
seqx=np.empty((0,20), float)
seqy=np.empty((0,1), float)


for i in range(100):
    filename='Text_Files_version4/txt/Activity_IMU_'+str(i+1)+'.txt'
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
        A.append(a)
    return np.array(A)

def seq2dataset2(seq,window):
    A=[]
    for i in range(len(seq)-window+1):
        a=seq[i:(i+window)]
        A.append(max(a)-1)
    return np.array(A)

w=100
X=seq2dataset1(seqx,w)
Y=seq2dataset2(seqy,w)


split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]


k=5
threshold = 0.5

f1_scores=[]
for train_index,val_index in KFold(k).split(x_train):
    xtrain,xval=x_train[train_index],x_train[val_index]
    ytrain,yval=y_train[train_index],y_train[val_index]

    model = Sequential()
    model.add(LSTM(units=128,return_sequences=True,input_shape=(100,20)))
    model.add(LSTM(units=64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    hist=model.fit(xtrain, ytrain, epochs=1, batch_size=100, validation_data=(x_test,y_test),verbose=2)
    accuracy=model.evaluate(xval,yval,verbose=0)[1]
    pred = model.predict(xval)
    pred_class = np.where(pred > threshold, 1, 0)


    """
    x_range=range(len(yval))
    plt.plot(x_range, yval[x_range], color='red')
    plt.plot(x_range, pred_class[x_range], color='blue')
    plt.grid()
    plt.show()
    """



    f1_scores.append(f1_score(yval, pred_class))

print("f1_score: ", f1_scores)
print(np.mean(f1_scores))
    
 

