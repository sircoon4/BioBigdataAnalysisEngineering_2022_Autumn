from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import csv

data = np.empty((0,23))
target = np.array([])
with open("FallData_Three.csv", "r") as f:

    lines = csv.reader(f)
    for line in lines:
        x = list(map(float, line[:-1]))
        y = int(line[-1])
        data = np.append(data, np.array([x]), axis=0)
        target = np.append(target, np.array([y]))
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)

#hyperparameter --> n_neighbors = 1, 5, 50, 100, 500
neigh = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cosine', metric_params=None, n_jobs=None, n_neighbors=1, p=2, weights='uniform')
neigh.fit(x_train, y_train)

y_pred = neigh.predict(x_test)

matrix = confusion_matrix(y_test, y_pred)
accuracy = (matrix[0,0] + matrix[1,1])/(matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1])
print(matrix)
print("Accuracy: ", accuracy)