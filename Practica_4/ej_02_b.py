#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 07-10-2020
File: ej_02.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import seaborn as snn

snn.set(font_scale=1)

from tensorflow.keras.datasets import cifar10
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Hago el flatten de los datos
x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
y_train = y_train.reshape(y_train.size)
y_test = y_test.reshape(y_test.size)

# def kk(y):
#     res = np.zeros(shape=(y.shape[0], 10))
#     res[np.arange(y.shape[0]), y] = 1
#     return res

# y_train = kk(y_train)
# y_test = kk(y_test)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Resto la media y divido por sigma
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma  # x_train /= 255
x_test = x_test - media
x_test /= sigma

# Input
inputs = keras.layers.Input(shape=(x_train.shape[1], ), name="Input")

l1 = keras.layers.Dense(100,
                        name='Hidden_1',
                        activation=keras.activations.relu,
                        use_bias=True,
                        kernel_regularizer=keras.regularizers.l2(5e-3),
                        kernel_initializer=keras.initializers.RandomUniform(
                            minval=-5e-2, maxval=5e-2, seed=None))(inputs)

output = keras.layers.Dense(
    10,
    name='Output',
    use_bias=True,
    activation=keras.activations.linear,
    kernel_regularizer=keras.regularizers.l2(5e-3),
    kernel_initializer=keras.initializers.RandomUniform(minval=-5e-2,
                                                        maxval=5e-2,
                                                        seed=None))(l1)

model = keras.models.Model(inputs=inputs,
                           outputs=output,
                           name='Ejercicio_3_4_o_5')

optimizer = keras.optimizers.SGD(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    # loss=keras.losses.categorical_crossentropy,
    # loss='categorical_crossentropy',
    # loss='mse',
    #   loss=keras.losses.binary_crossentropy,
    #   metrics=[keras.metrics.BinaryAccuracy()])
    #   metrics=[keras.metrics.Accuracy()])
    metrics=['accuracy'])
#   loss=keras.losses.categorical_crossentropy,
# metrics=keras.metrics.Accuracy())

model.summary()

history = model.fit(x_train,
                    y_train,
                    epochs=300,
                    validation_data=(x_test, y_test),
                    batch_size=50,
                    verbose=2)

y_pred = model.predict(x_test)

model.save('Modelo_ej2_Relu_Lin_Softmax.h5')
np.save("History_2_b_Softmax.npy", history.history)

plt.plot(history.history['loss'], label="Loss Training")
plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
# plt.axis("equal")
plt.savefig("Figuras/2_Loss_Softmax.png", format="png", bbox_inches="tight")
plt.show()

plt.plot(history.history['accuracy'], label="Acc. Training")
plt.plot(history.history['val_accuracy'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
# plt.axis("equal")
plt.savefig("Figuras/2_Acc_Softmax.png", format="png", bbox_inches="tight")
plt.show()