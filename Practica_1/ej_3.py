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
from keras.datasets import cifar10
from keras.datasets import mnist

import seaborn as snn

snn.set(font_scale = 1)


class KNN():

    def __init__(self, k=1, orden=2):
        self.k = k
        self.orden = orden
    
    def norma(self,x1, x2, o):
        return np.linalg.norm(x1 - x2, ord=o)
    
    def saveData(self, dataX, dataY, c_to_float=False):
        if(c_to_float):
            self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.float)
        else:
            self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.int16)
        self.Y = dataY.reshape(dataY.size)
    
    def predict(self,X, c_to_float=False):

        if(c_to_float):
            X = X.reshape(len(X), X[0].size).astype(np.float)
        else:
            X = X.reshape(len(X), X[0].size).astype(np.int16)
        #X = X.reshape(len(X), X[0].size).astype(np.int16)

        y_pred = np.array([])
        # y_pred = [self._predict(x) for x in X]        # Hago el for en vez de esto xq quiero ver cuanto avanza
       
        for i in range(len(X)):
            if (((i % 1000) == 0) and len(X) >= 500):
                print(i, "/", len(X))
            y_pred = np.append(y_pred, self.predictOne(X[i]) )

        #return np.array(y_pred)
        return y_pred
    
    def predictOne(self,x):
        distances = [ self.norma(x, datax, self.orden) for datax in self.X]

        index = np.argsort(distances)[:self.k]          # Me quedo con los k indices con menor distancia

        labels = np.array([self.Y[i] for i in index])    # Y me fijo de que clase son

        #return np.bincount(labels).argmax()             # No se que hace si empata, yo solo confio en Numpy
        most_common = Counter(labels).most_common(1)     # Busco el mas repetido
        return most_common[0][0]



def accuracy(y_true, y_pred):
    y_true = y_true.reshape(y_true.size)
    y_pred = y_pred.reshape(y_pred.size)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy*100


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

knn_cifar = KNN()
knn_cifar.saveData(x_train, y_train)
y_predict = knn_cifar.predict(x_test)
print("Cifar10 - Precision para norma 2 y k = 1: ", accuracy(y_predict, y_test) )



(x_train, y_train), (x_test, y_test) = mnist.load_data()

knn_mnist = KNN()
knn_mnist.saveData(x_train, y_train)
y_predict = knn_mnist.predict(x_test)
print("MNIST - Precision para norma 2 y k = 1: ", accuracy(y_predict, y_test) )



def ejercicio_3_20test(c='mnist',k=1,norma=2):
    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    knn = KNN(k,norma)

    knn.saveData(x_train, y_train)

    predict = knn.predict(x_test[:20])

    print(c," - Precision para norma", norma, "y k =", k, ":", accuracy(predict, y_test[:20]) )





"""
# Esto es para encontrar parametros buenos muy a ojo
N = 10
for norm in range(1,5):
    for k_v in range(1,10):

        knn = KNN(k_v, norm)
        knn.saveData(x_train, y_train)

        y_result = knn.predict(x_test[:N])

        print("Precision para norma", norm, "y k =", k_v, ":", accuracy(y_result, y_test[:N]) )
"""
