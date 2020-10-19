#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 09-10-2020
File: ej_03.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://towardsdatascience.com/batch-normalization-in-practice-an-example-with-keras-and-tensorflow-2-0-b1ec28bde96f
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, drop_arg, epochs, batch_size, nn, description

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers

# importo los datos
dim = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

# Muchos datos de test, prefiero dividirlo en proporciones distintas
x, y = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x,
                                                  y,
                                                  test_size=0.25,
                                                  stratify=y)

# Esto no hace falta, era para pasar a texto la reseña
indexes = imdb.get_word_index()
r_indexes = dict([(val, key) for key, val in indexes.items()])


# Funcion que Vectoriza datos teniendo en cuenta repeticiones
def vectorizeWCounts(x, dim):
    res = np.zeros((len(x), dim))
    for i, sequence in enumerate(x):
        values, counts = np.unique(sequence, return_counts=True)
        res[i, values] = counts
    return res


# Vectorizo los datos
x_train_v = vectorizeWCounts(x_train, dim)
x_test_v = vectorizeWCounts(x_test, dim)
x_val_v = vectorizeWCounts(x_val, dim)
y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)
y_val = y_val.astype(np.float)

# Arquitectura con dropout
inputs = layers.Input(shape=(x_train_v.shape[1], ), name="Input")

l1 = layers.Dense(nn, activation=activations.relu, name="Hidden_1")(inputs)

bn_1 = layers.BatchNormalization()(l1)

l2 = layers.Dense(nn, activation=activations.relu, name="Hidden_2")(bn_1)

bn_2 = layers.BatchNormalization()(l2)

outputs = layers.Dense(1, activation=activations.linear, name="Output")(bn_2)

model = keras.models.Model(inputs=inputs,
                           outputs=outputs,
                           name="Ejercicio_3_BN")

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[metrics.BinaryAccuracy(name='B_Acc')])

model.summary()

# Entreno
hist = model.fit(x_train_v,
                 y_train,
                 validation_data=(x_val_v, y_val),
                 epochs=epochs,
                 batch_size=batch_size,
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test_v, y_test)

# Guardo los datos
data_folder = os.path.join('Datos', '3_BN')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '3_BN')
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

plt.plot(hist.history['B_Acc'], label="Acc. Training")
plt.plot(hist.history['val_B_Acc'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}".format(test_Acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()