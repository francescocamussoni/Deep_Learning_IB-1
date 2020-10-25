#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 10-10-2020
File: ej_06.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, nn, rf, embedding_dim, epochs, drop_arg, batch_size, description

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers

# Cargo los datos
# Probar esto desde el cluster. Edit: Parece que funciona
path_folder = os.path.join("/", "share", "apps", "DeepLearning", "Datos")
file = "pima-indians-diabetes.csv"
path_file = os.path.join(path_folder, file)

# path_file = '/run/user/1000/gvfs/sftp:host=10.73.25.223,user=facundo.cabrera/share/apps/DeepLearning/Datos/pima-indians-diabetes.csv'

data = np.loadtxt(path_file, delimiter=',')

x = data[:, :-1]
y = data[:, -1].reshape((data.shape[0], 1))

# Arquitectura de la red
model = keras.models.Sequential(name='Ejercicio_6')

model.add(
    layers.Dense(nn,
                 input_shape=(x.shape[1], ),
                 activation='relu',
                 kernel_regularizer=regularizers.l2(rf),
                 name="Hidden_1"))

model.add(
    layers.Dense(nn,
                 activation='relu',
                 kernel_regularizer=regularizers.l2(rf),
                 name="Hidden_2"))

model.add(layers.Dense(1, activation=activations.linear, name="Output"))

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[metrics.BinaryAccuracy(name='Acc')])

model.summary()

# Guardo los pesos para cargarlos y "ressetear" el modelo en cada fold
data_folder = os.path.join('Datos', '6')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
sv = os.path.join(data_folder, "modelo_SIN_entrenar.h5")
model.save_weights(sv)

# 5-folding de los datos
kf = KFold(n_splits=5, shuffle=True)

# idx = np.arange(20)
idx = np.arange(x.shape[0])
# check = np.array([])

acc_test = np.array([])
acc_train = np.array([])
loss_test = np.array([])
loss_train = np.array([])

for train_index, test_index in kf.split(idx):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Cargo los pesos del modelo sin entrenar
    model.load_weights(sv)

    # Entreno
    hist = model.fit(x_train,
                     y_train,
                     validation_data=(x_test, y_test),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=2)

    acc_test = np.concatenate((acc_test, hist.history['val_Acc']))
    acc_train = np.concatenate((acc_train, hist.history['Acc']))

    loss_test = np.concatenate((loss_test, hist.history['loss']))
    loss_train = np.concatenate((loss_train, hist.history['val_loss']))

# os.remove(sv)     # esto me genera problemas. No se bien porque

acc_test = acc_test.reshape((5, epochs))
acc_train = acc_train.reshape((5, epochs))
loss_test = loss_test.reshape((5, epochs))
loss_train = loss_train.reshape((5, epochs))


def toDictionary(va, a, vl, l):

    res = {
        'test_min': va.min(axis=0),
        'test_max': va.max(axis=0),
        'test_mean': va.mean(axis=0),
        'train_min': a.min(axis=0),
        'train_max': a.max(axis=0),
        'train_mean': a.mean(axis=0),
        'ltest_min': vl.min(axis=0),
        'ltest_max': vl.max(axis=0),
        'ltest_mean': vl.mean(axis=0),
        'ltrain_min': l.min(axis=0),
        'ltrain_max': l.max(axis=0),
        'ltrain_mean': l.mean(axis=0)
    }

    return res


results = toDictionary(acc_test, acc_train, loss_test, loss_train)
np.save(os.path.join(data_folder, '{}.npy'.format(description)), results)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '6')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

xx = np.arange(epochs)
plt.fill_between(xx, results['train_min'], results['train_max'], alpha=0.35)
plt.plot(results['train_mean'])
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
# plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# Grafico
plt.fill_between(xx, results['test_min'], results['test_max'], alpha=0.35)
plt.plot(results['test_mean'])
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Test Accuracy", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_Test_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

plt.fill_between(xx, results['ltrain_min'], results['ltrain_max'], alpha=0.35)
plt.plot(results['ltrain_mean'], label="Loss Training")
plt.fill_between(xx, results['ltest_min'], results['ltest_max'], alpha=0.35)
plt.plot(results['ltest_mean'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Loss_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

# check.sort()    # Si todo esta bien esto deberia tener todos los enteros de 0 a x.shape sin repetir
