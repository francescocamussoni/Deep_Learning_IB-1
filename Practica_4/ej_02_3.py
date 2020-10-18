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
from CLArg import lr  

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

