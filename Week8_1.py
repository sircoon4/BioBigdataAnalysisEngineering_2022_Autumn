# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:16:46 2022

@author: sirco
"""

import openpyxl
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pydotplus # pip install pydotplus
from IPython.display import display, Image


import graphviz # conda install graphviz

loc = "."
file = (loc+'/PlayTennis.csv')
tennis = pd.read_csv(file)
tennis
# 데이터 전처리 과정: 문자열(string) 타입 -> 숫자(int) 타입
tennis.Outlook = tennis.Outlook.replace('Sunny', 0)
tennis.Outlook = tennis.Outlook.replace('Overcast', 1)
tennis.Outlook = tennis.Outlook.replace('Rain', 2)
tennis.Temperature = tennis.Temperature.replace('Hot', 3)
tennis.Temperature = tennis.Temperature.replace('Mild', 4)
tennis.Temperature = tennis.Temperature.replace('Cool', 5)
tennis.Humidity = tennis.Humidity.replace('High', 6)
tennis.Humidity = tennis.Humidity.replace('Normal', 7)
tennis.Wind = tennis.Wind.replace('Weak', 8)
tennis.Wind = tennis.Wind.replace('Strong', 9)
tennis.PlayTennis = tennis.PlayTennis.replace('No', 10)
tennis.PlayTennis = tennis.PlayTennis.replace('Yes', 11)
tennis
# 입력 칼럼들(속성)과 출력 칼럼(클래스)을 labeling
X = np.array(pd.DataFrame(tennis, columns = ['Outlook', 'Temperature',
'Humidity', 'Wind']))
Y = np.array(pd.DataFrame(tennis, columns = ['PlayTennis']))
print("X:\n", X)
print("Y:\n", Y)
#훈련 데이터(10개)와 테스트 데이터(4개) 분리
#여기서는편의상 ‘train_test_split’ 모듈 사용 안 함
X_train = X[0:10]
Y_train = Y[0:10]
X_test = X[10:14]
Y_test = Y[10:14]
DT = DecisionTreeClassifier(criterion='entropy')
DT = DT.fit(X_train, Y_train)
DT_prediction = DT.predict(X_test)
print("Desirable Result = \n", Y_test)
print("Prediction Result = \n", DT_prediction)

#’tennis’ 데이터의 각 컬럼명을 list 형태로 변환 후 저장
#’[0:4], 즉 [Outlook, Temp., Humidity, Wind] 슬라이싱 후 저장
feature_names = tennis.columns.tolist()
feature_names = feature_names[0:4]
target_name = np.array(['Play No', 'Play Yes'])
DT_dot_data = tree.export_graphviz(DT, out_file = None,
feature_names = feature_names,
class_names = target_name,
filled = True, rounded = True,
special_characters = True)
#그래프 로드 및 이미지파일(png)로 생성
DT_graph = pydotplus.graph_from_dot_data(DT_dot_data)
display(Image(DT_graph.create_png()))
