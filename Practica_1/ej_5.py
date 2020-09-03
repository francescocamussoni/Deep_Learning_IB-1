#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 02-09-2020
File: ej_5.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.datasets import mnist

np.random.seed(10)

class LinearClassifier():
    def __init__(self, n):
        self.n = n
        pass
    def fit(self):
        pass
    def predict(self):
        pass
    def loss_gradient(self):
        pass


class SVM(LinearClassifier):
    def __init__(self, n, delta=1):
        super(SVM, self).__init__(n)
        self.delta = delta

# Numero de clases
n = 10
n_samples = 4

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X = x_train[:n_samples]
Y = y_train[:n_samples]

delta = 1
landa = 1

#X.shape = (#ejemplos, dim(x)+1)
#W.shape = (#clases, dim(x)+1)
#Scores = np.dot(X,W.T)


X = X.reshape(len(X), X[0].size)        # Reacomodo la entrada

X = np.hstack((np.ones((len(X),1)), X))  # Pongo 1 para el bias

W = np.random.uniform(0, 1, size=(n, X.shape[1]))   # Invento la W

Scores = np.dot(W, X.T)         #Calculo los Scores


# Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)

# El primer indice es el Score que debe ganar y el segundo el numero de imagen
idx = np.arange(0,n_samples)

Y = Y.reshape(Y.size)           # Aca tuve que reacomodar Y porque tiene un formato feo

# Creo que con esto deberia poder quedarme con los scores que deben ganar
y_win = Scores[Y, idx]


# Intento restar cada score a su columna (ie imagen) correspondiente

resta = Scores - y_win[np.newaxis,:] + delta

resta = np.maximum(resta, 0)        # Ahora mato todo lo que no sea positivo


resta[Y, idx] = 0           # y acomodo los 0 de los que deberian ganar
print(resta)

resta[resta>0] = 1

# ME FALTA SUMAR TODA UNA COLUMNA Y PONERLA EN EL LUGAR QUE TIENE QUE GANAR!!!!!!!!!!!!!!!!!!!!

resta[Y, idx] -= resta.sum(axis=0)[idx]



dw = (np.dot(resta, X) / n ) + landa * W






"""
Metrica accuracy
"""