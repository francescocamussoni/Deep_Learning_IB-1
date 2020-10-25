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
from CLArg import lr, rf, epochs, batch_size, description

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

model.add(layers.Conv2D(32, 3, (2, 2), activation='relu', padding='same'))
model.add(layers.Conv2D(64, 3, (2, 2), activation='relu', padding='same'))
model.add(layers.Conv2D(128, 3, (2, 2), activation='relu', padding='valid'))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu', name='embedded'))
model.add(layers.Dense(1152, activation='relu'))

model.add(layers.Reshape((3, 3, 128)))

model.add(
    layers.Conv2DTranspose(64, 3, (2, 2), activation='relu', padding='valid'))
model.add(
    layers.Conv2DTranspose(32, 3, (2, 2), activation='relu', padding='same'))
model.add(
    layers.Conv2DTranspose(1, 3, (2, 2), activation='sigmoid', padding='same'))

model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    # loss=losses.MeanSquaredError(name='loss'),
    loss=losses.BinaryCrossentropy(name='loss'),
    metrics=[
        metrics.BinaryAccuracy(name='B_Acc'),
        metrics.MeanSquaredError(name='MSE'),
    ])

# Entreno
hist = model.fit(x_train_n,
                 x_train,
                 validation_data=(x_val_n, x_val),
                 epochs=epochs,
                 batch_size=batch_size,
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc, test_MSE = model.evaluate(x_test_n, x_test)

# Guardo los datos
data_folder = os.path.join('Datos', '7_Internet_BCE')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Tomo un modelo auxiliar para ver el output del encoder
embedded = keras.models.Model(model.input, model.get_layer('embedded').output)

# Tomo un ejemplo para graficar
eg = np.random.randint(0, x_test_n.shape[0])

# # Guardo las imagenes
img_folder = os.path.join('Figuras', '7_Internet_BCE')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
colums = 10
for i in range(colums):
    ax = plt.subplot(4, colums, i + 1)
    ax.imshow(x_train[eg + i].reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, colums, i + 1 + colums)
    ax.imshow(x_train_n[eg + i].reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

    o_embedded = embedded.predict(x_train_n[eg + i].reshape(1, 28, 28, 1))

    ax = plt.subplot(4, colums, i + 1 + 2 * colums)
    ax.imshow(o_embedded.reshape(10, 1), cmap='Greys_r')
    ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

    predict = model.predict(x_train_n[eg + i].reshape(1, 28, 28, 1))

    ax = plt.subplot(4, colums, i + 1 + 3 * colums)
    ax.imshow(predict.reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Graf_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# Loss
plt.plot(hist.history['loss'], label="Loss Training")
plt.plot(hist.history['val_loss'], label="Loss Validation")
plt.title("Acc Test: {:.3f}   MSE: {:.3f}".format(test_Acc, test_MSE))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Loss_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# Binary Accuracy
plt.plot(hist.history['B_Acc'], label="Acc. Training")
plt.plot(hist.history['val_B_Acc'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}   MSE: {:.3f}".format(test_Acc, test_MSE))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# MSE
plt.plot(hist.history['MSE'], label="MSE Training")
plt.plot(hist.history['val_MSE'], label="MSE Validation")
plt.title("Acc Test: {:.3f}   MSE: {:.3f}".format(test_Acc, test_MSE))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'MSE_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()