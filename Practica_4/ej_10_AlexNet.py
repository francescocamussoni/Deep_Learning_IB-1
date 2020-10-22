#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-10-2020
File: ej_10_AlexNet.py
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

# Script propio para pasar argumentos por linea de comandos
from CLArg import lr, rf, epochs, batch_size, drop_arg, description
from CLArg import dataset

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10, cifar100

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

# Importo los datos
if dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif dataset == 'cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Los junto porque quiero splitearlos distinto
x, y = np.vstack((x_train, x_test)), np.hstack((y_train, y_test))
# Separo los datos de test
x, x_test, y, y_test = train_test_split(x, y, test_size=9000, stratify=y)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x,
                                                  y,
                                                  test_size=9000,
                                                  stratify=y)