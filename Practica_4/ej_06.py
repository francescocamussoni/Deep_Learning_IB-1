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
    "-bz",
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
kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']
drop_arg = kwargs['Dropout']

print("-------------------------------------")
print('lr: {} rf: {} do: {} epochs: {} bs: {}'.format(lr, rf, drop_arg, epochs,
                                                      batch_size))
print("-------------------------------------")

s
# Cargo los datos
# Probar esto desde el cluster
# path_folder = os.path.join("share","apps","DeepLearning","Datos")
# file = "pima-indians-diabetes.csv"
# path_file = os.path.join(path_folder, file)

path_file = '/run/user/1000/gvfs/sftp:host=10.73.25.223,user=facundo.cabrera/share/apps/DeepLearning/Datos/pima-indians-diabetes.csv'

data = np.loadtxt(path_file, delimiter=',')

x = data[:,:-1]
y = data[:,-1].reshape((data.shape[0],1))



inputs = layers.Input(shape=(x.shape[1], ), name="Input")

layer_1 = layers.Dense(10,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_1")(inputs)

layer_2 = layers.Dense(10,
                       activation=activations.relu,
                       use_bias=True,
                       kernel_regularizer=regularizers.l2(rf),
                       name="Hidden_2")(layer_1)

outputs = layers.Dense(1,
                       activation=activations.linear,
                       use_bias=True,
                       name="Output")(layer_2)

model = keras.models.Model(inputs=inputs,
                           outputs=outputs,
                           name="Ejercicio_6")

model.compile(optimizer=optimizers.SGD(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
              metrics=[
                  metrics.BinaryAccuracy(name='Acc')
              ])

model.summary()












# 5-folding de los datos
kf = KFold(n_splits=5, shuffle=True)

# idx = np.arange(20)
idx = np.arange(x.shape[0])

# check = np.array([])

for train_index, test_index in kf.split(idx):
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print("x_train: ", x_train)
    # print("x_test : ", x_test)
    # check = np.append(check, x_test)

# check.sort()
