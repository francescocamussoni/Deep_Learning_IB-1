#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 10-10-2020
File: ej_05.py
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
    default=0,
    help="Regularizer factor (default: 0)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=200,
    help="Epochs (default: 200)",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=None,
    help="Batch size (default: None)",
)
parser.add_argument(
    "-do",
    "--Dropout",
    type=float,
    default=1,
    help="Dropout argument (default: 0)",
)
kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']
drop_arg = kwargs['Dropout']

print('lr: {} rf: {} do: {} epochs: {} bs: {}'.format(lr, rf, drop_arg, epochs,
                                                      batch_size))

# Datos
nData = 100000
x = np.linspace(0, 1, nData).reshape((nData, 1))
y = 4 * x * (1 - x)

idx = np.arange(nData)
np.random.shuffle(idx)

x_train = x[idx[:-len(idx) // 5]]
x_test = x[idx[-len(idx) // 5:]]

y_train = x[idx[:-len(idx) // 5]]
y_test = x[idx[-len(idx) // 5:]]

# Arquitectura de la red
inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

layer_1 = layers.Dense(5,
                       activation=activations.tanh,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name='Hidden')(inputs)

concat = layers.Concatenate()([inputs, layer_1])

outputs = layers.Dense(1,
                       activation=activations.linear,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name='Output')(concat)

model = keras.models.Model(inputs=inputs, outputs=outputs, name='Ejercicio_5')

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.MeanSquaredError(name='loss'),
              metrics=[metrics.MeanSquaredError(name='acc_MSE')])

model.summary()

# Entreno
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

# Guardo los datos
data_folder = os.path.join('Datos', '5')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(
    os.path.join(
        data_folder,
        'lr={}_rf={}_do={}_e={}_bs={}.h5'.format(lr, rf, drop_arg, epochs,
                                                 batch_size)))
np.save(
    os.path.join(
        data_folder,
        'lr={}_rf={}_do={}_e={}_bs={}.npy'.format(lr, rf, drop_arg, epochs,
                                                  batch_size)),
    history.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '5')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.plot(history.history['loss'], label="Loss Training")
plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Loss_lr={}_rf={}_do={}_e={}_bs={}.png'.format(lr, rf, drop_arg, epochs,
                                                   batch_size)),
            format="png",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['acc_MSE'], label="Acc. Training")
plt.plot(history.history['val_acc_MSE'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder, 'Acc_Dropout_lr={}_rf={}_do={}_e={}_bs={}.png'.format(
        lr, rf, drop_arg, epochs, batch_size)),
            format="png",
            bbox_inches="tight")
plt.close()