#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 07-10-2020
File: ej_02_3.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, batch_size

from tensorflow.keras.datasets import cifar10

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    activations,
    regularizers,
    losses,
    metrics,
    optimizers,
)

# Cargo los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Hago el flatten de los datos
x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)

# Paso los labels a una matriz binaria
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalizacion
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
x_test = x_test - media
x_test /= sigma

# Arquitectura de la red segun el ej3 TP2
inputs = keras.layers.Input(shape=x_train.shape[1] , name="Input")

l1 = keras.layers.Dense(
    100,
    name='Hidden',
    activation=keras.activations.sigmoid,
    kernel_regularizer=keras.regularizers.l2(rf),
)(inputs)

output = layers.Dense(10,
                      name='Output',
                      activation=activations.linear,
                      kernel_regularizer=regularizers.l2(rf))(l1)

model = keras.models.Model(inputs=inputs, outputs=output, name='Ej3_TP2')

model.compile(optimizer=keras.optimizers.Adam(lr),
              loss=losses.mean_squared_error,
              metrics=[metrics.CategoricalAccuracy(name="Acc")])

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    # validation_data=(x_test, y_test),  # XXX
    batch_size=batch_size,
    verbose=2)

y_pred = model.predict(x_test)
