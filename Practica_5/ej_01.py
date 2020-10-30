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
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from utils import lr, rf, epochs, batch_size, description
from utils import small_dataset

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#small_dataset = True
#path_data = "/home/cabre1994/Desktop/Deep_Learning/Deep_Learning_IB/Datasets"
path_data = os.getcwd()

if small_dataset:
    save_images = os.path.join(path_data, "dogs.vs-cats_small.npy")
    save_label = os.path.join(path_data, "dogs.vs-cats_small_label.npy")
    path_data = os.path.join(path_data, "dogs-vs-cats_small")
else:
    save_images = os.path.join(path_data, "dogs.vs-cats.npy")
    save_label = os.path.join(path_data, "dogs.vs-cats_label.npy")
    path_data = os.path.join(path_data, "dogs-vs-cats")

i = 1

if not os.path.exists(save_images):
    images = np.array([],dtype=np.uint8)
    labels = np.array([],dtype=np.uint8)

    for file in os.listdir(path_data):
        print(i)
        i += 1
        #if i > 100:
        #    continue
        
        if small_dataset:
            img = load_img(os.path.join(path_data,file))
        else:
            img = load_img(os.path.join(path_data,file), target_size=(299,299,3),interpolation="bilinear")

        img_arr = img_to_array(img).astype(np.uint8)

        images = np.append(images, img_arr)

        if file.startswith('cat'):
            labels = np.append(labels, 0)
        else:
            labels = np.append(labels, 1)

    if small_dataset:
        images = images.reshape(-1,32,32,3)
        np.save(save_images, images)
        np.save(save_label, labels)
    else:
        images = images.reshape(-1,299,299,3)
        np.save(save_images, images)
        np.save(save_label, labels)

# Importo los datos
x_train = np.load(save_images)
y_train = np.load(save_label)

# Separo los datos de test
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=4000,
                                                    stratify=y_train)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=4000,
                                                  stratify=y_train)

# Normalizacion
media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
x_test = x_test - media
x_test /= sigma
x_val = x_val - media
x_val /= sigma

# Paso los labels a one-hot encoded
#y_train = keras.utils.to_categorical(y_train, n_classes)
#y_test = keras.utils.to_categorical(y_test, n_classes)
#y_val = keras.utils.to_categorical(y_val, n_classes)


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
