#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 21-10-2020
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

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, embedding_dim, epochs, drop_arg, batch_size, description

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers

# Importo los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Separo el train en train y validation
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=1/6,
                                                  stratify=y_train)

# Concateno los datos para que sea mas facil normalizar y agregar ruido
# x, y = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
# x, y = np.vstack((x_train, x_test)), np.hstack((y_train, y_test))

# Normalizo
x_train = x_train / 255
x_test = x_test / 255
x_val = x_val / 255
# Agrego ruido Gaussiano con std = 0.5
x_train_n = x_train + np.random.normal(scale=0.5, size=x_train.shape)
x_test_n = x_test + np.random.normal(scale=0.5, size=x_test.shape)
x_val_n = x_val + np.random.normal(scale=0.5, size=x_val.shape)
# Satuo los valores en el rango [0,1]
np.clip(x_train_n, 0, 1, out=x_train_n)
np.clip(x_test_n, 0, 1, out=x_test_n)
np.clip(x_val_n, 0, 1, out=x_val_n)






# Separo el train en train y validation
# x_train, x_val, y_train, y_val = train_test_split(x_train,
#                                                   y_train,
#                                                   test_size=1/6,
#                                                   stratify=y_train)

# Normalizacion y ruido

# Arquitecura con Convolucionales









# TODO Cambiar a test 
n = 5
f = plt.figure(figsize=(10, 4))
for i in range(n):
    # display noisy
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + 1+ n)
    plt.imshow(x_train_n[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(3, n, i + 1 + 2*n)
#     plt.imshow(y_test_noisy[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# f.savefig('denoisingDigits_dense.png', bbox_inches='tight')
plt.show()