#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-10-2020
File: ej_10_AlexNet.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, batch_size, drop_arg, description
from CLArg import dataset

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, cifar100

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importo los datos
if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    n_classes = 10
elif dataset == 'cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    n_classes = 100

# Los junto porque quiero splitearlos distinto
x, y = np.vstack((x_train, x_test)), np.vstack((y_train, y_test))
# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=9000, stratify=y)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x,
                                                  y,
                                                  test_size=9000,
                                                  stratify=y)

# Normalizacion
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
x_test = x_test - media
x_test /= sigma
x_val = x_val - media
x_val /= sigma

# Arquitectura de la mini-AlexNet
model = keras.models.Sequential(name='Mini-AlexNet')

model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(64, 5, strides=2, activation='relu', padding='valid'))
# model.add(layers.Conv2D(32,5,strides=2,activation='relu',padding='same'))
model.add(layers.MaxPool2D(3, strides=1))
# model.add(layers.MaxPool2D(2,strides=1))

# model.add(layers.Conv2D(192,5,strides=1,activation='relu',padding='same'))
model.add(layers.Conv2D(192, 5, strides=1, activation='relu', padding='valid'))
model.add(layers.MaxPool2D(3, strides=1))

model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))
model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))
model.add(layers.Conv2D(192, 3, strides=1, activation='relu', padding='same'))
model.add(layers.MaxPool2D(3, strides=1))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='linear'))

model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=[metrics.CategoricalAccuracy(name='CAcc')])





# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_31 (Conv2D)           (None, 55, 55, 96)        34944
# _________________________________________________________________
# max_pooling2d_16 (MaxPooling (None, 27, 27, 96)        0
# _________________________________________________________________
# conv2d_32 (Conv2D)           (None, 27, 27, 256)       614656
# _________________________________________________________________
# max_pooling2d_17 (MaxPooling (None, 13, 13, 256)       0
# _________________________________________________________________
# conv2d_33 (Conv2D)           (None, 13, 13, 384)       885120
# _________________________________________________________________
# conv2d_34 (Conv2D)           (None, 13, 13, 384)       1327488
# _________________________________________________________________
# conv2d_35 (Conv2D)           (None, 13, 13, 256)       884992
# _________________________________________________________________
# max_pooling2d_18 (MaxPooling (None, 6, 6, 256)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 9216)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 4096)              37752832
# _________________________________________________________________
# dense_2 (Dense)              (None, 4096)              16781312
# _________________________________________________________________
# dense_3 (Dense)              (None, 1000)              4097000
# =================================================================
# Total params: 62,378,344
# Trainable params: 62,378,344
# Non-trainable params: 0
# _________________________________________________________________

# # Arquitectura de la AlexNet - Original
# model = keras.models.Sequential(name='AlexNet')

# model.add(layers.Input(shape=(227,227,3)))

# model.add(layers.Conv2D(96,11,strides=4,activation='relu'))
# model.add(layers.MaxPool2D(3,strides=2))

# model.add(layers.Conv2D(256,5,strides=1,activation='relu',padding='same'))
# model.add(layers.MaxPool2D(3,strides=2))

# model.add(layers.Conv2D(384,3,strides=1,activation='relu',padding='same'))
# model.add(layers.Conv2D(384,3,strides=1,activation='relu',padding='same'))
# model.add(layers.Conv2D(256,3,strides=1,activation='relu',padding='same'))
# model.add(layers.MaxPool2D(3,strides=2))

# model.add(layers.Flatten())
# model.add(layers.Dense(4096,activation='relu'))
# model.add(layers.Dense(4096,activation='relu'))
# model.add(layers.Dense(1000,activation='softmax'))

# Ratio entre num. parametros y dim de entrada
# 62,378,344 / (227*227*3) = 403.5
# 403 * (32*32*3) = 1238016