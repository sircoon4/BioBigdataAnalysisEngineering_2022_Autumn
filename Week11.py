# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import csv
data = np.empty((0,23))
target = np.array([])
with open("FallData_Two.csv", "r") as f:

    lines = csv.reader(f)
    for line in lines:
        x = list(map(float, line[:-1]))
        y = int(line[-1])
        data = np.append(data, np.array([x]), axis=0)
        target = np.append(target, np.array([y]))

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)

x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = np.array(y_train)-1
y_test = np.array(y_test)-1
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

# 신경망 구조
n_input = 23
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_hidden4=512
n_output=2


dmlp_ce = Sequential()
dmlp_ce.add(Dense(units=n_hidden1,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',
               bias_initializer='zeros'))
dmlp_ce.add(Dense(units=n_hidden2,activation='tanh',kernel_initializer='random_uniform',
               bias_initializer='zeros'))
dmlp_ce.add(Dense(units=n_hidden3,activation='tanh',kernel_initializer='random_uniform',
               bias_initializer='zeros'))
dmlp_ce.add(Dense(units=n_hidden4,activation='tanh',kernel_initializer='random_uniform',
               bias_initializer='zeros'))
dmlp_ce.add(Dense(units=n_output,activation='softmax',kernel_initializer='random_uniform',
               bias_initializer='zeros'))
dmlp_ce.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.1),metrics=['accuracy'])
history_ce = dmlp_ce.fit(x_train,y_train,batch_size=128,epochs=100,validation_data = (x_test,y_test),verbose=2)

result_ce = dmlp_ce.evaluate(x_test,y_test,verbose=2)
print("정확도",result_ce[1]*100)

import matplotlib.pyplot as plt

#정확도 곡선
plt.plot(history_ce.history['accuracy'])
plt.plot(history_ce.history['val_accuracy'])
plt.title('Model accuracy for ce')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['train_ce','validation_ce'],loc='best')
plt.grid()
plt.show()


