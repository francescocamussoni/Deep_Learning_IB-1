#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 30-08-2020
File: ej_3.py
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
from keras.datasets import mnist

def norma(x1, x2, o):
    #kk = np.concatenate(x1-x2, axis=0)
    #kkk = np.concatenate(kk, axis=0)
    #return np.linalg.norm(kkk, ord=int(orden))
    #return np.linalg.norm(kkk, ord=int(o))
    #return np.linalg.norm(x1 - x2)
    return np.linalg.norm(x1 - x2, ord=o)
# import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

class KNN():

    def __init__(self, k=10, orden=2):
        self.k = k
        self.orden = orden
    
    def saveData(self, dataX, dataY):
        #self.X = dataX.reshape(dataX.size)
        self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.int16)
        #self.X = dataX
        self.Y = dataY
    
    def predict2(self,X):
        X = X.reshape(len(X), X[0].size).astype(np.int16)
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


(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
knn = KNN()
knn.saveData(x_train, y_train)
y_result = knn.predict(x_test[:N])
print("Presicion: ", accuracy(y_result, y_test[:N][:,0]) )
"""
"""
N = 20
for norm in range(1,5):
    for k_v in range(1,21):

        knn = KNN(k_v, norm)
        knn.saveData(x_train, y_train)

        y_result = knn.predict2(x_test[:N])

        print("Precision para norma ", norm, "y k = ", k_v, ": ", accuracy(y_result, y_test[:N]) )      # Aca tambien cambie algo
"""

N = 10000

knn = KNN(1, 2)
knn.saveData(x_train, y_train)

y_result = knn.predict2(x_test[:N])

print("Precision para norma ", 1, "y k = ", 2, ": ", accuracy(y_result, y_test[:N]) )

"""
knn = KNN(20)
knn.saveData(x_train, y_train)

y_result = knn.predict(x_test[:N])

print("Presicion para norma ", 2, "y k = ", k_v, ": ", accuracy(y_result, y_test[:N][:,0]) )
"""
