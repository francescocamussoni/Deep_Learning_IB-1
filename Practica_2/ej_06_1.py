#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 21-09-2020
File: ej_06.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from myModules import models, layers, regularizers, losses, activations, metrics, optimizers

# x = np.random.randint(0,10, size=(5,10))

# inputt = layers.Input(x.shape[1])

# model = models.Network(inputt)

# layer1 = layers.Dense(units=7)
# layer2 = layers.Dense(units=8)
# # layer3 = layers.ConcatInput(inputt)
# layer3 = layers.Concat(inputt, model.forward, 0)
# # layer3 = layers.Concat(layer1, model.forward, 1)
# layer4 = layers.Dense(units=50)

# model.add(layer1)
# model.add(layer2)
# model.add(layer3)
# model.add(layer4)

x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train = np.array([[1],[-1],[-1],[1]])
y_train = y_train.reshape(y_train.size,1)

reg1 = regularizers.L2(0)
reg2 = regularizers.L1(0)

inputt = layers.Input(x_train.shape[1])

model = models.Network(inputt)

# model.add(layers.Dense(units=2,activations.Tanh(),input_dim=x_train.shape[1], regularizer=reg1))
# model.add(layers.Dense(units=1,activations.Tanh(), regularizer=reg2))

model.add(layers.Dense(units=2, activation=activations.Tanh(), regu=reg1, w=1))
model.add(layers.Dense(units=1, activation=activations.Tanh(), regu=reg2, w=1))

# import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

model.fit(x=x_train, y=y_train, bs= x_train.shape[0], epochs=10000,
            opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR, plot=False)

"""
# Identify the problem: 'XOR' or 'Image'
problem_flag = 'XOR' #P6
# Dataset
x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train = np.array([[1],[-1],[-1],[1]])
# Regularizer
reg1 = regularizers.L2(0.1)
reg2 = regularizers.L1(0.2)
# Create model
model = models.Network()
model.add(layers.Dense(units=2,activations.Tanh(),input_dim=x_train.shape[1], regularizer=reg1))
model.add(layers.Dense(units=1,activations.Tanh(), regularizer=reg2))
# Train network
model.fit(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
batch_size=x_train.shape[0], epochs=200, opt=optimizers.SGD(lr=0.05),
problem_flag=problem_flag)
"""