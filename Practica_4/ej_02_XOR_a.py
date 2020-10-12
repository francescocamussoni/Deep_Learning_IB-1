#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 09-10-2020
File: ej_02_XOR_a.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics, layers, optimizers, losses, activations

# Datos que no son datos
x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_train = np.array([[1], [-1], [-1], [1]])
y_train = y_train.reshape(y_train.size, 1)

# Arquitectura de la red
inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

layer_1 = layers.Dense(2,
                       name='Hidden_1',
                       activation=activations.tanh,
                       use_bias=True)(inputs)

outputs = layers.Dense(1,
                       name='Output',
                       activation=activations.tanh,
                       use_bias=True)(layer_1)

model = keras.models.Model(inputs=inputs,
                           outputs=outputs,
                           name='XOR_Arquitectura_1')


def my_acc(y_true, y_pred):
    acc = tf.reduce_mean(
        tf.cast(tf.less_equal(tf.abs(y_true - y_pred), 0.1), tf.float32))
    return acc


model.compile(optimizer=optimizers.SGD(learning_rate=1e-2),
              loss=losses.MSE,
              metrics=[my_acc])

model.summary()

history = model.fit(x_train, y_train, epochs=10000, verbose=2)

# Guardo los datos
data_folder = os.path.join('Datos', '2')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '2_model_XOR_A.h5'))
np.save(os.path.join(data_folder, '2_history_XOR_A.npy'), history.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '2')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.plot(history.history['loss'], label="Loss")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '2_Loss_XOR_A.png'),
            format="png",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['my_acc'], label="Accuracy")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
# plt.axis("equal")
plt.savefig(os.path.join(img_folder, '2_Acc_XOR_A.png'),
            format="png",
            bbox_inches="tight")
plt.close()