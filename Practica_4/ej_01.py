#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 07-10-2020
File: ej_01.py
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

from sklearn.datasets import load_boston
from tensorflow import keras

boston = load_boston()
# print(boston.DESCR) # Para printear la descripcion del dataset

dim = len(boston["feature_names"])

idx = np.arange(506)
np.random.shuffle(idx)

x_train = boston["data"][idx[:-len(idx) // 4]]
x_test = boston["data"][idx[-len(idx) // 4:]]

y_train = boston["target"][idx[:-len(idx) // 4]]
y_test = boston["target"][idx[-len(idx) // 4:]]

media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma  # x_train /= 255
x_test = x_test - media
x_test /= sigma

# plt.plot(x_train, y_train, '.b')
# plt.plot(x_test,  y_test,  '.r')
# plt.show()

# Con model
inputs = keras.layers.Input(shape=x_train.shape[1], name="Input")
ol = keras.layers.Dense(1, name="Output")(inputs)

model = keras.models.Model(inputs=inputs, outputs=ol, name="LinearRegression")

model.summary()

optimizer = keras.optimizers.SGD(learning_rate=1e-3)

model.compile(optimizer, loss=keras.losses.MSE, metrics=["mse"])

history = model.fit(x_train,
                    y_train,
                    epochs=200,
                    validation_data=(x_test, y_test),
                    verbose=2)

y_pred = model.predict(x_test)

plt.plot(y_test, y_pred, "ob", label="Predicciones")
plt.plot(y_test, y_test, "k", label="Target")
plt.xlabel("Precios reales [k$]",fontsize=15)
plt.ylabel("Precios predichos [k$]",fontsize=15)
# plt.axis("equal")
plt.savefig("Figuras/1.png", format="png", bbox_inches="tight")
plt.show()

plt.plot(history.history['loss'], label="Loss Training")
plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
# plt.axis("equal")
plt.savefig("Figuras/2_Loss.png", format="png", bbox_inches="tight")
plt.show()

plt.plot(history.history['mse'], label="Acc. Training")
plt.plot(history.history['val_mse'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
# plt.axis("equal")
plt.savefig("Figuras/2_Acc.png", format="png", bbox_inches="tight")
plt.show()

# Con Sequential, para practicar

# from tensorflow.keras.utils import plot_model

# plot_model(model, show_shapes=True, to_file="model.png")
