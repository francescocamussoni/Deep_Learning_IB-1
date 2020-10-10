#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 09-10-2020
File: ej_03.py
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import (
    layers,
    activations,
    regularizers,
    losses,
    metrics,
    optimizers,
)

# Argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate (default: 1e-3)",
)
parser.add_argument(
    "-rf",
    "--regularizer_factor",
    type=float,
    default=1e-1,
    help="Regularizer factor (default: 1e-1)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=200,
    help="Epochs (default: 200)",
)
parser.add_argument(
    "-bz",
    "--batch_size",
    type=int,
    default=None,
    help="Batch size (default: None)",
)
kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']

# importo los datos
dim = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

indexes = imdb.get_word_index()
r_indexes = dict([(val, key) for key, val in indexes.items()])

# def vectorize(x, dim):
#     res = np.zeros((len(x), dim))
#     for i, sequence in enumerate(x):
#         values, counts = np.unique(sequence, return_counts=True)
#         res[i, values] = 1
#     return res


def vectorize(x, dim):
    res = np.zeros((len(x), dim))
    for i, sequence in enumerate(x):
        res[i, sequence] = 1
    return res


def vectorizeWCounts(x, dim):
    res = np.zeros((len(x), dim))
    for i, sequence in enumerate(x):
        values, counts = np.unique(sequence, return_counts=True)
        res[i, values] = counts
    return res


x_train_v = vectorizeWCounts(x_train, dim)
x_test_v = vectorizeWCounts(x_test, dim)
y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)
# x_v2 = vectorize(x_train, dim)

# Arquitectura con regularizadores
inputs = layers.Input(shape=(x_train_v.shape[1], ), name="Input")

layer_1 = layers.Dense(25,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_1")(inputs)

layer_2 = layers.Dense(25,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_2")(layer_1)

outputs = layers.Dense(1,
                       activation=activations.linear,
                       use_bias=True,
                       name="Output")(layer_2)

model = keras.models.Model(inputs=inputs,
                           outputs=outputs,
                           name="Ejercicio_3_Regularizadores")

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[
                  metrics.BinaryAccuracy(name='B_Acc'),
                  metrics.Accuracy(name='Acc')
              ])

model.summary()

# Entreno
history = model.fit(x_train_v,
                    y_train,
                    validation_data=(x_test_v, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

# Guardo los datos
data_folder = os.path.join('Datos', '3')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(
    os.path.join(
        data_folder,
        'Regul_lr={}_rf={}_e={}_bs={}.h5'.format(lr, rf, epochs,
                                                   batch_size)))
np.save(    os.path.join(
        data_folder,
        'Regul_lr={}_rf={}_e={}_bs={}.npy'.format(lr, rf, epochs,
                                                    batch_size)),
        history.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '3')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.plot(history.history['loss'], label="Loss")
plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Loss_Regul_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs, batch_size)),
            format="pdf",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['Acc'], label="Acc. Training")
plt.plot(history.history['val_Acc'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Acc_Regul_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs, batch_size)),
            format="pdf",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['B_Acc'], label="Acc. Training")
plt.plot(history.history['val_B_Acc'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'B_Acc_Regul_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs, batch_size)),
            format="pdf",
            bbox_inches="tight")
plt.close()