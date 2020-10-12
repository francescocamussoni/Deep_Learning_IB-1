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

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

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
parser.add_argument(
    "-nn",
    "--NumNeuronas",
    type=int,
    default=10,
    help="Numero de neuronas (default: 10)",
)
kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']
drop_arg = kwargs['Dropout']
nn = kwargs['NumNeuronas']

print("-------------------------------------")
print('lr: {} rf: {} do: {} epochs: {} bs: {} nn: {}'.format(lr, rf, drop_arg, epochs,
                                                      batch_size, nn))
print("-------------------------------------")

# Cargo los datos
# Probar esto desde el cluster. Edit: Parece que funciona
path_folder = os.path.join("/","share","apps","DeepLearning","Datos")
file = "pima-indians-diabetes.csv"
path_file = os.path.join(path_folder, file)

# path_file = '/run/user/1000/gvfs/sftp:host=10.73.25.223,user=facundo.cabrera/share/apps/DeepLearning/Datos/pima-indians-diabetes.csv'

data = np.loadtxt(path_file, delimiter=',')

x = data[:, :-1]
y = data[:, -1].reshape((data.shape[0], 1))

inputs = layers.Input(shape=(x.shape[1], ), name="Input")

layer_1 = layers.Dense(nn,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_1")(inputs)

layer_2 = layers.Dense(nn,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_2")(layer_1)

outputs = layers.Dense(1,
                       activation=activations.linear,
                       use_bias=True,
                       name="Output")(layer_2)

model = keras.models.Model(inputs=inputs, outputs=outputs, name="Ejercicio_6")

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[metrics.BinaryAccuracy(name='Acc')])

model.summary()

# Guardo los pesos para cargarlos y "ressetear" el modelo en cada fold
data_folder = os.path.join('Datos', '6')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
save_weights = os.path.join(data_folder,"modelo_SIN_entrenar.h5")
model.save_weights(save_weights)

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
    model.load_weights(save_weights)

    # Entreno
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2)
    
    acc_test = np.concatenate((acc_test, history.history['val_Acc'] ))
    acc_train = np.concatenate((acc_train, history.history['Acc'] ))

    loss_test = np.concatenate((loss_test, history.history['loss'] ))
    loss_train = np.concatenate((loss_train, history.history['val_loss'] ))

    # acc_test = np.vstack( (acc_test, history.history['val_Acc']) )


    # kkparachequear = idx[test_index]
    # check = np.append(check, kkparachequear)

os.remove(save_weights)

acc_test = acc_test.reshape( (5, epochs) )
acc_train = acc_train.reshape( (5, epochs) )
loss_test = acc_train.reshape( (5, epochs) )
loss_train = acc_train.reshape( (5, epochs) )

def toDictionary(va, a, vl, l):

    res ={'test_min' : va.min(axis=0),
    'test_max' : va.max(axis=0),
    'test_mean' : va.mean(axis=0),

    'train_min' : a.min(axis=0),
    'train_max' : a.max(axis=0),
    'train_mean' : a.mean(axis=0),

    'ltest_min' : vl.min(axis=0),
    'ltest_max' : vl.max(axis=0),
    'ltest_mean' : vl.mean(axis=0),

    'ltrain_min' : l.min(axis=0),
    'ltrain_max' : l.max(axis=0),
    'ltrain_mean' : l.mean(axis=0)}

    return res

results = toDictionary(acc_test, acc_train, loss_test, loss_train)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '6')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

xx = np.arange(epochs)
plt.fill_between(xx, results['train_min'], results['train_max'], alpha=0.2)
plt.plot(results['train_mean'], '-')
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
# plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Acc_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs,
                                                   batch_size)),
            format="pdf",
            bbox_inches="tight")
# plt.show()
plt.close()

# Grafico
plt.fill_between(xx, results['test_min'], results['test_max'], alpha=0.2)
plt.plot(results['test_mean'], '-')
# plt.plot(history.history['loss'], label="Loss")
# plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Test Accuracy", fontsize=15)
# plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Acc_Test_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs,
                                                   batch_size)),
            format="pdf",
            bbox_inches="tight")
# plt.show()
plt.close()

plt.fill_between(xx, results['ltrain_min'], results['ltrain_max'], alpha=0.2)
plt.plot(results['ltrain_mean'], '-', label="Loss")
plt.fill_between(xx, results['ltest_min'], results['ltest_max'], alpha=0.2)
plt.plot(results['ltest_mean'], '-', label="Loss Test")
# plt.plot(history.history['loss'], label="Loss")
# plt.plot(history.history['val_loss'], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(
    img_folder,
    'Loss_Regul_lr={}_rf={}_e={}_bs={}.pdf'.format(lr, rf, epochs,
                                                   batch_size)),
            format="pdf",
            bbox_inches="tight")
# plt.show()
plt.close()

# check.sort()    # Si todo esta bien esto deberia tener todos los enteros de 0 a x.shape sin repetir
