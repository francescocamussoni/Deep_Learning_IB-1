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
                                                  test_size=1 / 6,
                                                  stratify=y_train)

# Reshape de la entrada para Convolucionales y normalizacion
x_train = x_train.reshape((-1, 28, 28, 1)) / 255
x_test = x_test.reshape((-1, 28, 28, 1)) / 255
x_val = x_val.reshape((-1, 28, 28, 1)) / 255
# Agrego ruido Gaussiano con std = 0.5
x_train_n = x_train + np.random.normal(scale=0.5, size=x_train.shape)
x_test_n = x_test + np.random.normal(scale=0.5, size=x_test.shape)
x_val_n = x_val + np.random.normal(scale=0.5, size=x_val.shape)
# Satuo los valores en el rango [0,1]
x_train_n = np.clip(x_train_n, 0, 1)
x_test_n = np.clip(x_test_n, 0, 1)
x_val_n = np.clip(x_val_n, 0, 1)

# Arquitecura con Convolucionales
model = keras.models.Sequential(name='Autoencoder_Conv')

model.add(layers.Input(shape=(28, 28, 1)))

# Encoder
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(16, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(8, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D(padding='same', name='Hola'))

# model.add(layers.BatchNormalization())

# Decoder
model.add(layers.Conv2D(8, 3, activation=activations.relu, padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(16, 3, activation=activations.relu, padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='valid'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(1, 3, activation=activations.linear, padding='same'))

model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
    metrics=[metrics.BinaryAccuracy(name='B_Acc'), "accuracy", "mse"])

model.predict(x_train[:1])

# TODO Cambiar a test
n = 1
f = plt.figure(figsize=(10, 4))
for i in range(n):
    # display noisy
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_train_n[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    kk = model.get_layer('Hola').output

    import ipdb
    ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(kk[0].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# f.savefig('denoisingDigits_dense.png', bbox_inches='tight')
plt.show()