# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:44:56 2022

@author: sirco
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os

cnn=tf.keras.models.load_model("my_cnn_for_deploy.h5")
class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'flog', 'horse', 'ship', 'truck']

x_test=[]
for filename in os.listdir("./test_images"):
    if 'jpg' not in filename:
        continue
    img=Image.open("./test_images/"+filename)
    x=np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
x_test=np.asarray(x_test)

pred=cnn.predict(x_test)

import matplotlib.pyplot as plt

n=len(x_test)
plt.figure(figsize=(18,4))

for i in range(n):
    plt.subplot(2,n,i+1)
    plt.imshow(x_test[i])
    plt.xticks([]);plt.yticks([])
    plt.subplot(2,n,n+i+1)
    if i==0:
        plt.barh(class_names,pred[i])
    else:
        plt.barh(['a','A', 'b', 'c', 'd', 'D', 'f', 'h', 's', 't'],pred[i])
    plt.xticks([])