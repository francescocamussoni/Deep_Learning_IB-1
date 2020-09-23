#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-09-2020
File: ej_08.py
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

import seaborn as snn
snn.set(font_scale = 1.1)

from myModules import models, layers, regularizers, losses, activations, metrics, optimizers

np.random.seed(14)  # Con MSE, landa = 1e-2, lr=1e-3, con std y media, 1e-2 las 2 1ras w


# Cargo los datos de cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Hago el flatten de los datos
x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
y_train = y_train.reshape(y_train.size)
y_test = y_test.reshape(y_test.size)

# Resto la media y divido por sigma
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train  = x_train - media
x_train /= sigma    # x_train /= 255
x_test   = x_test  - media
x_test  /= sigma

n, dim = x_train.shape  # Numero de ejemplos de training y dimension del problema


reg1 = regularizers.L2(1e-2)
reg2 = regularizers.L2(1e-2)
reg3 = regularizers.L2(1e-2)


inputt = layers.Input(x_train.shape[1])

model = models.Network(inputt)

# Capa Oculta 1
model.add(layers.Dense(units=100, activation=activations.Relu(), regu=reg1, w=1e-2))
# Capa Oculta 2
model.add(layers.Dense(units=100, activation=activations.Relu(), regu=reg2, w=1e-2))
# Capa de Salida
model.add(layers.Dense(units=10, activation=activations.Linear(), regu=reg3, w=1e-1))

# import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

model.fit(x=x_train, y=y_train, bs= 50, epochs=300, x_test=x_test, y_test=y_test,
            opt=optimizers.SGD(lr=1e-3), loss=losses.MSE(), metric=metrics.accuracy, plot=True)