# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:55:13 2022

@author: sirco
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

batch_siz=128
n_epoch=10
k=5

def cross_validation(data_gen, dropout_rate, l1_reg):
    accuracy=[]
    for train_index, val_index in KFold(k).split(x_train):
        xtrain, xval=x_train[train_index],x_train[val_index]
        ytrain, yval=y_train[train_index],y_train[val_index]
        
        cnn=Sequential()
        cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
        cnn.add(Conv2D(32,(3,3),activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Dropout(dropout_rate[0]))
        cnn.add(Conv2D(64,(3,3),activation='relu'))
        cnn.add(Conv2D(64,(3,3),activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Dropout(dropout_rate[1]))
        cnn.add(Flatten())
        cnn.add(Dense(512,activation='relu'))
        cnn.add(Dropout(dropout_rate[2]))
        cnn.add(Dense(10,activation='softmax',kernel_regularizer=regularizers.L1(l1_reg)))
        
        cnn.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
        if data_gen:
            generator = ImageDataGenerator(rotation_range=3.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
            cnn.fit_generator(generator.flow(x_train, y_train, batch_size=batch_siz),epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)
        else:
            cnn.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)
        accuracy.append(cnn.evaluate(xval,yval,verbose=0)[1])
    return accuracy

acc_000=cross_validation(False, [0.0,0.0,0.0], 0.0)
acc_001=cross_validation(False, [0.0,0.0,0.0], 0.01)
acc_010=cross_validation(False, [0.25,0.25,0.5], 0.0)
acc_011=cross_validation(False, [0.25,0.25,0.5], 0.01)
acc_100=cross_validation(True, [0.0,0.0,0.0], 0.0)
acc_101=cross_validation(True, [0.0,0.0,0.0], 0.01)
acc_110=cross_validation(True, [0.25,0.25,0.5], 0.0)
acc_111=cross_validation(True, [0.25,0.25,0.5], 0.01)

print("출력 형식: [Data augmentation-Dropout-l2 regularizer] (교차검증 시도/평균)")
print("[000] (",acc_000,"/",np.array(acc_000).mean(),")")
print("[001] (",acc_001,"/",np.array(acc_001).mean(),")")
print("[010] (",acc_010,"/",np.array(acc_010).mean(),")")
print("[011] (",acc_011,"/",np.array(acc_011).mean(),")")
print("[100] (",acc_100,"/",np.array(acc_100).mean(),")")
print("[101] (",acc_101,"/",np.array(acc_101).mean(),")")
print("[110] (",acc_110,"/",np.array(acc_110).mean(),")")
print("[111] (",acc_111,"/",np.array(acc_111).mean(),")")

import matplotlib.pyplot as plt

plt.grid()
plt.boxplot([acc_000,acc_001,acc_010,acc_011,acc_100,acc_101,acc_110,acc_111], labels=["000","001","010","011","100","101","110","111"])

        