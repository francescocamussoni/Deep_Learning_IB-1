#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 01-09-2020
File: ej_4.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
#from keras.datasets import cifar10
#from keras.datasets import mnist

def norma(x1, x2, o):
    return np.linalg.norm(x1 - x2, ord=o)

class KNN():

    def __init__(self, k=1, orden=2):
        self.k = k
        self.orden = orden
    
    def saveData(self, dataX, dataY):
        #self.X = dataX.reshape(dataX.size)
        #self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.int16)
        self.X = dataX.reshape(len(dataX), dataX[0].size)
        #self.X = dataX
        self.Y = dataY
    
    def predict2(self,X):
        #X = X.reshape(len(X), X[0].size).astype(np.int16)
        X = X.reshape(len(X), X[0].size)
        y_pred = np.array([])
        for i in range(len(X)):
            if ((i % 100) == 0):
                print(i, "/", len(X))
            y_pred = np.append(y_pred, self._predict(X[i]) )
        return y_pred
    
    def _predict2(self,x):
        distances = [ norma(x, datax, self.orden) for datax in self.X]

        k_idx = np.argsort(distances)[:self.k]

        k_neighbor_labels = np.array([self.Y[i] for i in k_idx])

        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

        most_common = Counter(k_neighbor_labels[:,0]).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        y_pred = np.array([])
        for i in range(len(X)):
            y_pred = np.append(y_pred, self._predict(X[i]) )
        #y_pred = [self._predict(x) for x in X]
        #return np.array(y_pred)
        return y_pred
    
    def _predict(self, x):
        # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
        # print("KK")
        distances = [ norma(x, datax, self.orden) for datax in self.X]

        k_idx = np.argsort(distances)[:self.k]

        k_neighbor_labels = np.array([self.Y[i] for i in k_idx])

        # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

        most_common = Counter(k_neighbor_labels).most_common(1)        # Aca Cambie algo
        return most_common[0][0]


def accuracy(y_true, y_pred):
    accuracy = float(np.sum(y_true == y_pred)) / float(len(y_true))
    return accuracy


class createData():
    def __init__(self, mean_, std_, label):
        self.mean = mean_
        self.std = std_
        self.label = label
    
    def getPoints(self, n):
        data = np.random.normal(self.mean, self.std, size=(n,2))
        label = np.full(n, self.label)
        return data, label


clstr1 = createData([0,0], 1, 0)
clstr2 = createData([3,3], 1, 1)
clstr3 = createData([3,-3], 1, 2)
clstr4 = createData([-3,-3], 1, 3)
clstr5 = createData([-3,3], 1, 4)

train_1, predict_1 = clstr1.getPoints(100)
train_2, predict_2 = clstr2.getPoints(100)
train_3, predict_3 = clstr3.getPoints(100)
train_4, predict_4 = clstr4.getPoints(100)
train_5, predict_5 = clstr5.getPoints(100)

plt.scatter(train_1[:,0], train_1[:,1])
plt.scatter(train_2[:,0], train_2[:,1])
plt.scatter(train_3[:,0], train_3[:,1])
plt.scatter(train_4[:,0], train_4[:,1])
plt.scatter(train_5[:,0], train_5[:,1])
plt.show()

train = np.concatenate([train_1, train_2, train_3, train_4, train_5])
predict = np.concatenate([predict_1, predict_2, predict_3, predict_4, predict_5])


knn = KNN(7,2)

knn.saveData(train, predict)


# Ahora genero datos de test
test_1, check_1 = clstr1.getPoints(50)
test_2, check_2 = clstr2.getPoints(50)
test_3, check_3 = clstr3.getPoints(50)
test_4, check_4 = clstr4.getPoints(50)
test_5, check_5 = clstr5.getPoints(50)

test = np.concatenate([test_1, test_2, test_3, test_4, test_5])
check = np.concatenate([check_1, check_2, check_3, check_4, check_5])

result = knn.predict(test)

print("Precision para norma ", 2, "y k = ", 1, ": ", accuracy(result, check) )



#####
# Grafico

steps = 300

xmin, xmax = train[:,0].min() - 1, train[:,0].max() + 1
ymin, ymax = train[:,1].min() - 1, train[:,1].max() + 1
x_span = np.linspace(xmin, xmax, steps)
y_span = np.linspace(ymin, ymax, steps)
xx, yy = np.meshgrid(x_span, y_span)


labels = knn.predict2(np.c_[xx.ravel(), yy.ravel()])

z = labels.reshape(xx.shape)



#colores = ['g', 'b', 'r', 'm', 'y', 'b']

colores = np.linspace(0,1,5)

cols = [colores[i] for i in predict]
cols2 = [colores[i] for i in predict]

plt.figure()
plt.axes([0.04, 0.04, 0.99, 0.99])

plt.contourf(xx, yy, z, 5, alpha=.3, cmap='jet')

#plt.scatter(train[:,0], train[:,1], c=predict, cmap='jet', alpha=0.6)
plt.scatter(test[:,0], test[:,1], c=check, marker='x', cmap='jet', alpha=0.6)

plt.show()

"""

Algunas cosas para mejorar

- Poner con X los datos test, la X con el color que deberia tener 
- El predict tiene un 2
- Hacer una funcion mejor que lo del common 
"""
