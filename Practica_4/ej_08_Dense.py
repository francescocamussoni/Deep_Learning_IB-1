#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 21-10-2020
File: ej_08.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://tensorchiefs.github.io/dl_course/exercises/07_fcn_MNIST_tricks
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, batch_size, drop_arg, description

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

# Importo los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Los junto porque creo que no estan bien distribuidos
x, y = np.vstack((x_train, x_test)), np.hstack((y_train, y_test))
# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=1 / 7, stratify=y)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x,
                                                  y,
                                                  test_size=1 / 6,
                                                  stratify=y)

# Normalizacion
media = x_train.mean(axis=0)

x_train = x_train - media
x_train = x_train / 255
x_test = x_test - media
x_test = x_test / 255
x_val = x_val - media
x_val = x_val / 255

# Paso los labels a one-hot representation
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
y_val = keras.utils.to_categorical(y_val, 10)

# Arquitectura de la red con capas densas
model = keras.models.Sequential(name='MNIST_Dense')

model.add(layers.Flatten(input_shape=x_train.shape[1:]))
model.add(layers.Dense(500, 'relu', kernel_regularizer=l2(rf)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_arg))
model.add(layers.Dense(50, 'relu', kernel_regularizer=l2(rf)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_arg))
model.add(layers.Dense(10, 'linear', kernel_regularizer=l2(rf)))

model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=[metrics.CategoricalAccuracy(name='CAcc')])

hist = model.fit(x_train,
                 y_train,
                 epochs=epochs,
                 validation_data=(x_val, y_val),
                 batch_size=batch_size,
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test, y_test)

# Guardo los datos
data_folder = os.path.join('Datos', '8_Dense')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '8_Dense')
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