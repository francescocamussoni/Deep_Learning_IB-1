#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 28-10-2020
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: ssh facundo.cabrera@rocks7frontend.fisica.cabib
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification#Import-Library
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from utils import lr, rf, epochs, batch_size, description
from utils import dogs_cats

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path del cluster
path_data = os.path.join("/", "share", "apps", "DeepLearning", "Datos")

# Path local
path_data = os.getcwd()

path_file = os.path.join(path_data, "dogs-vs-cats")

files = os.listdir(path_file)
labels = np.array([],dtype=np.int)

# Armo los labels
for file in files:
    if file.startswith('cat'):
        labels = np.append(labels, 0)
    else:
        labels = np.append(labels, 1)

train = pd.DataFrame({'files':files,'labels':labels})

train["labels"] = train["labels"].replace({0: 'cat', 1: 'dog'})


# Arquitectura de la mini-VGG16
model = keras.models.Sequential(name='Mini-VGG16')

model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(64, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, 3, strides=1, activation='relu', padding='same'))

model.add(layers.MaxPool2D(2, strides=2))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, 3, strides=1, activation='relu', padding='same'))

model.add(layers.MaxPool2D(2, strides=2))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, 3, strides=1, activation='relu', padding='same'))

model.add(layers.MaxPool2D(2, strides=1))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))

model.add(layers.MaxPool2D(2, strides=1))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, 3, strides=1, activation='relu', padding='same'))

model.add(layers.MaxPool2D(2, strides=1))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024, activation='relu', kernel_regularizer=l2(rf)))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(rf)))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(rf)))
model.add(layers.Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=[metrics.BinaryAccuracy(name='CAcc')])


# Callbacks
earlystop = keras.callbacks.EarlyStopping(patience=10)
lrr = keras.callbacks.ReduceLROnPlateau('val_acc',0.1,2,1,min_lr=1e-5)
callbacks = [earlystop, lrr]

# Separo entre train y test
train, test = train_test_split(train, test_size=4000, stratify=train['labels'])
# Ahora separo entre training y validacion
train, val = train_test_split(train, test_size=4000, stratify=train['labels'])

# Borrar
#FAST_RUN = False
#IMAGE_WIDTH=128
#IMAGE_HEIGHT=128
#IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
#IMAGE_CHANNELS=3

train_IDG = ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=5,
    height_shift_range=5
)

train_generator = train_IDG.flow_from_dataframe(
    train,
    path_file,
    x_col='files',
    y_col='labels',
    target_size=(32,32),
    class_mode='binary',
    batch_size=batch_size
)

val_IDG = ImageDataGenerator(rescale=1./255)
validation_generator = val_IDG.flow_from_dataframe(
    val,
    path_file,
    x_col='files',
    y_col='labels',
    target_size=(32,32),
    class_mode='binary',
    batch_size=batch_size
)

#example_df = train.sample(n=1).reset_index(drop=True)
#example_generator = train_IDG.flow_from_dataframe(
#    example_df,
#    path_file,
#    x_col='files',
#    y_col='labels',
#    target_size=(32,32),
#    class_mode='categorical'
#)
#
#
#plt.figure(figsize=(12, 12))
#for i in range(0, 15):
#    plt.subplot(5, 3, i+1)
#    for X_batch, Y_batch in example_generator:
#        image = X_batch[0]
#        plt.imshow(image)
#        break
#plt.tight_layout()
#plt.show()




import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT


IDG = ImageDataGenerator(
    rotation_range=45,  # Ang max de rotaciones
    width_shift_range=5,    # Cant de pixeles que puede trasladarse, sepuede pasar una
    height_shift_range=5,   # fraccion de la dimension en vez de un entero
    shear_range=0.,     # No entendi que es
    zoom_range=0.,      # Por lo que vi, queda re feo asi que no lo uso
    fill_mode='nearest',    # Estrategia para llenar los huecos
    horizontal_flip=True,   # Reflexion horizontal b -> d
    vertical_flip=False,    # Reflexion vertical   ! -> ¡
    # Con esto alcanza creo, el resto no tengo tan claro como funciona
    # y prefiero dejarlo asi
)

# Only required if featurewise_center or featurewise_std_normalization
# or zca_whitening are set to True.
# IDG.fit(x_train)

hist = model.fit(IDG.flow(x_train, y_train, batch_size=batch_size),
                 epochs=epochs,
                 steps_per_epoch=len(x_train) / batch_size,
                 validation_data=(x_val, y_val),
                 verbose=2)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_Acc = model.evaluate(x_test, y_test)

data_folder = os.path.join('Datos', '1')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', '1')
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

plt.plot(hist.history['CAcc'], label="Acc. Training")
plt.plot(hist.history['val_CAcc'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}".format(test_Acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()
