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
https://machinelearningmastery.com/how
-to-configure-image-data-augmentation-when-training
-deep-learning-neural-networks/
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, batch_size, description
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
x_train, y_train = np.vstack((x_train, x_test)), np.vstack((y_train, y_test))
# Separo los datos de test
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=9000,
                                                    stratify=y_train)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=9000,
                                                  stratify=y_train)

# Normalizacion
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
x_test = x_test - media
x_test /= sigma
x_val = x_val - media
x_val /= sigma

# Paso los labels a one-hot encoded
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
y_val = keras.utils.to_categorical(y_val, n_classes)

# Arquitectura de la mini-AlexNet
model = keras.models.Sequential(name='Mini-AlexNet')

model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(96, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(rf*0.1)))
model.add(layers.MaxPool2D(3, strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(rf*0.1)))
model.add(layers.MaxPool2D(3, strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(384, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(rf)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(384, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(rf)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(rf*0.1)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='relu', kernel_regularizer=l2(rf)))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='relu', kernel_regularizer=l2(rf)))
model.add(layers.Dense(n_classes, activation='linear'))

model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=[metrics.CategoricalAccuracy(name='CAcc')])

IDG = ImageDataGenerator(
    # Ang max de rotaciones
    rotation_range=30,
    # Cant de pixeles que puede trasladarse, sepuede pasar una
    # fraccion de la dimension en vez de un entero
    width_shift_range=5,
    height_shift_range=5,
    brightness_range=[0.5, 1.5],  # Cuanto puede variar el brillo
    shear_range=0.,  # No entendi que es
    zoom_range=0.,  # Por lo que vi, queda re feo asi que no lo uso
    fill_mode='nearest',  # Estrategia para llenar los huecos
    horizontal_flip=True,  # Reflexion horizontal b -> d
    vertical_flip=True,  # Reflexion vertical ! -> ¡
    # Con esto alcanza creo, el resto no tengo tan claro como funciona
    # y prefiero dejarlo asi
)

# Only required if featurewise_center or featurewise_std_normalization
# or zca_whitening are set to True.
# IDG.fit(x_train)

# hist = model.fit(IDG.flow(x_train, y_train, batch_size=batch_size),
hist = model.fit_generator(IDG.flow(x_train, y_train, batch_size=batch_size),
                 epochs=epochs,
                 steps_per_epoch=len(x_train) / batch_size,
                 validation_data=(x_val, y_val),
                #  workers=4,
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test, y_test)

data_folder = os.path.join('Datos', '10_AlexNet' + dataset)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '10_AlexNet' + dataset)
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.plot(hist.history['loss'], label="Loss Training")
plt.plot(hist.history['val_loss'], label="Loss Validation")
plt.title("Acc Test: {:.3f}".format(test_Acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Loss_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

plt.plot(hist.history['CAcc'], label="Acc. Training")
plt.plot(hist.history['val_CAcc'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}".format(test_Acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

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