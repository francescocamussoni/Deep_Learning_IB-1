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

from sklearn.datasets import load_boston
from tensorflow import keras

import seaborn as snn
snn.set(font_scale = 1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

boston = load_boston()
# print(boston.DESCR) # Para printear la descripcion del dataset

dim = len(boston["feature_names"])

idx = np.arange(506)
np.random.shuffle(idx)

x_train = boston["data"][idx[:-len(idx) // 4]]
y_train = boston["target"][idx[:-len(idx) // 4]]
x_test = boston["data"][idx[-len(idx) // 4:]]
y_test = boston["target"][idx[-len(idx) // 4:]]

# Centramos y normalizamos los datos
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
x_test = x_test - media
x_test /= sigma

# Con models.Model
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

# Guardo los datos
data_folder = os.path.join('Datos', '1')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
model.save(os.path.join(data_folder, '1_model.h5'))
np.save(os.path.join(data_folder, '1_history.npy'), history.history)

# Prediccion para los datos de test
y_pred = model.predict(x_test)

# Grafico y guardo figuras
img_folder = os.path.join('Figuras', '1')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

plt.plot(y_test, y_pred, "ob", label="Predicciones")
plt.plot(y_test, y_test, "k", label="Target")
plt.xlabel("Precios reales [k$]", fontsize=15)
plt.ylabel("Precios predichos [k$]", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '1.pdf'),
            format="pdf",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['loss'], label="Loss Training")
plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '1_Loss.pdf'),
            format="pdf",
            bbox_inches="tight")
plt.close()

plt.plot(history.history['mse'], label="Acc. Training")
plt.plot(history.history['val_mse'], label="Acc. Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '1_Acc.pdf'),
            format="pdf",
            bbox_inches="tight")
plt.close()