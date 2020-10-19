#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-10-2020
File: ej_04_Conv1D.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, embedding_dim, epochs, drop_arg, batch_size, description

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
# Hacemos el padding ahora que estan todos los datos juntos
x = keras.preprocessing.sequence.pad_sequences(x,
                                               padding='post',
                                               dtype=np.float)
# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x,
                                                  y,
                                                  test_size=0.25,
                                                  stratify=y)

# Arquitecura con Convolucionales
inputs = layers.Input(shape=x.shape[1], name="Input")

embe = layers.Embedding(dim, embedding_dim, input_length=x.shape[1])(inputs)

drop_1 = layers.Dropout(drop_arg)(embe)

zp1 = layers.ZeroPadding1D()(drop_1)
l1 = layers.Conv1D(filters=32, kernel_size=3, activation=activations.relu)(zp1)
mp = layers.MaxPool1D()(l1)

drop_2 = layers.Dropout(drop_arg)(mp)

zp2 = layers.ZeroPadding1D()(drop_2)
l2 = layers.Conv1D(filters=64, kernel_size=3, activation=activations.relu)(zp2)

flatten = layers.Flatten()(l2)

drop_3 = layers.Dropout(drop_arg)(flatten)

outputs = layers.Dense(1, activation=activations.linear,
                       name="Output")(drop_3)

model = keras.models.Model(inputs=inputs,
                           outputs=outputs,
                           name="Ejercicio_4_Conv")

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[metrics.BinaryAccuracy(name='B_Acc')])

model.summary()


# Entreno
hist = model.fit(x_train,
                 y_train,
                 validation_data=(x_val, y_val),
                 epochs=epochs,
                 batch_size=batch_size,
                 verbose=1)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test, y_test)

# Guardo los datos
data_folder = os.path.join('Datos', '4_Conv')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '4_Conv')
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