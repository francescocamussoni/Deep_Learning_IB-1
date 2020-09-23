#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-09-2020
File: ej_07.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import itertools
from myModules import models, layers, regularizers, losses, activations, metrics, optimizers


N1 = 6
N2 = 100
cant_ejemplos = 2**N1

x_train = np.array([x for x in itertools.product([-1,1], repeat=N1)])
y_train = np.prod(x_train,axis=1).reshape(cant_ejemplos,1)

reg1 = regularizers.L2(0)
reg2 = regularizers.L1(0)


inputt = layers.Input(x_train.shape[1])

model = models.Network(inputt)

model.add(layers.Dense(units=N2, activation=activations.Tanh(), regu=reg1, w=1))
model.add(layers.Dense(units=1,  activation=activations.Tanh(), regu=reg2, w=1))

model.fit(x=x_train, y=y_train, bs= x_train.shape[0], epochs=40000,
            opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR, plot=False)