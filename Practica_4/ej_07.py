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
model.add(layers.Conv2D(64, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='same'))
model.add(layers.MaxPooling2D(padding='same', name='encoder'))

# Decoder
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(32, 3, activation=activations.relu, padding='same'))
model.add(layers.UpSampling2D())
# model.add(layers.Conv2D(64, 3, activation=activations.relu, padding='same'))
model.add(layers.Conv2D(64, 3, activation=activations.relu, padding='valid'))
model.add(layers.UpSampling2D())
# model.add(layers.BatchNormalization())
model.add(layers.Conv2D(1, 3, activation=activations.linear, padding='same'))
model.add(tf.keras.layers.ReLU(max_value=1))

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    loss=losses.MeanSquaredError(name='loss'),
    # loss=losses.BinaryCrossentropy(name='loss'),
    metrics=[
        metrics.BinaryAccuracy(name='B_Acc'),
        metrics.MeanSquaredError(name='MSE'),
    ])

# Entreno
hist = model.fit(
    x_train_n[:20],
    x_train[:20],
    #  validation_data=(x_val_n[:5], x_val[:5]),
    epochs=1000,
    batch_size=batch_size,
    verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc, test_MSE = model.evaluate(x_test_n, x_test)

# Guardo los datos
data_folder = os.path.join('Datos', '7')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Tomo un modelo auxiliar para ver el output del encoder
encoder = keras.models.Model(model.input, model.get_layer('encoder').output)

# Tomo un ejemplo para graficar
eg = np.random.randint(0, x_test_n.shape[0])

predict = model.predict(x_test_n[eg:eg + 1])
o_encoder = encoder.predict(x_test_n[eg:eg + 1])
o_encoder = o_encoder[0]

# Guardo las imagenes
img_folder = os.path.join('Figuras', '7')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
ax = plt.subplot(6, 8, 4)
ax.imshow(x_test[eg].reshape(28, 28),cmap='Greys_r')
# plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(6, 8, 5)
ax.imshow(x_test_n[eg].reshape(28, 28),cmap='Greys_r')
# plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
for j in range(1, 5):
    for i in range(8):
        ax = plt.subplot(6, 8, 8 * j + i + 1)
        ax.imshow(o_encoder[:, :, 8 * (j - 1) + i].reshape(4, 4),cmap='Greys_r')
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
ax = plt.subplot(6, 1, 6)
ax.imshow(predict.reshape(28, 28),cmap='Greys_r')
ax.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Graf_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# Grafico
plt.plot(hist.history['loss'], label="Loss Training")
plt.plot(hist.history['val_loss'], label="Loss Validation")
plt.title("Acc Test: {:.3f}\tMSE: {:.3f}".format(test_Acc, test_MSE))
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
plt.title("Acc Test: {:.3f}\tMSE: {:.3f}".format(test_Acc, test_MSE))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

plt.plot(hist.history['MSE'], label="MSE Training")
plt.plot(hist.history['val_MSE'], label="MSE Validation")
plt.title("Acc Test: {:.3f}\tMSE: {:.3f}".format(test_Acc, test_MSE))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'MSE_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()