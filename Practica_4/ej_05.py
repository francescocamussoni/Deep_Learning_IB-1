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

import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, drop_arg, batch_size, description

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers

# Datos
nData = 1000000
x = np.linspace(0, 1, nData).reshape((nData, 1))
y = 4 * x * (1 - x)

# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=0.1)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1 / 9)

# Arquitectura de la red
inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

layer_1 = layers.Dense(5,
                       activation=activations.tanh,
                       kernel_regularizer=regularizers.l2(rf),
                       name='Hidden')(inputs)

concat = layers.Concatenate()([inputs, layer_1])

outputs = layers.Dense(1,
                       activation=activations.linear,
                       kernel_regularizer=regularizers.l2(rf),
                       name='Output')(concat)

model = keras.models.Model(inputs=inputs, outputs=outputs, name='Ejercicio_5')

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.MeanSquaredError(name='loss'),
              metrics=[metrics.MeanSquaredError(name='acc_MSE')])

model.summary()

# Entreno
hist = model.fit(x_train,
                 y_train,
                 validation_data=(x_val, y_val),
                 epochs=epochs,
                 batch_size=batch_size,
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test, y_test)

# Guardo los datos
data_folder = os.path.join('Datos', '5')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '5')
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

plt.plot(hist.history['acc_MSE'], label="Acc. Training")
plt.plot(hist.history['val_acc_MSE'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}".format(test_Acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()