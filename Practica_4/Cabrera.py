#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-10-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import imdb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import losses, metrics, optimizers

import seaborn as snn
snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

#--------------------------------------
#           Ejercicio 1
#--------------------------------------


def ejercicio_1():
    from sklearn.datasets import load_boston

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

    model = keras.models.Model(inputs=inputs,
                               outputs=ol,
                               name="LinearRegression")

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
    img_folder = os.path.join("Informe", 'Figuras', '1')
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


#--------------------------------------
#           Ejercicio 2
#--------------------------------------


def ejercicio_2_EJ3TP2():
    lr = 1e-4
    rf = 1e-3
    batch_size = 128
    epochs = 100
    description = "lr={}_rf={}_bs={}".format(lr, rf, batch_size)
    # Cargo los datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # separo el training en train y validation manteniendo la distribucion (y mezclando)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      stratify=y_train)

    # Hago el flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    x_val = x_val.reshape(len(x_val), x_val[0].size).astype(np.float)

    # Paso los labels a one-hot encoded
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

    # Normalizacion
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train = x_train - media
    x_train /= sigma
    x_test = x_test - media
    x_test /= sigma
    x_val = x_val - media
    x_val /= sigma

    # Arquitectura de la red segun el ej3 TP2
    inputs = layers.Input(shape=x_train.shape[1], name="Input")

    l1 = layers.Dense(
        100,
        name='Hidden',
        activation=activations.sigmoid,
        kernel_regularizer=regularizers.l2(rf),
    )(inputs)

    output = layers.Dense(10,
                          name='Output',
                          activation=activations.linear,
                          kernel_regularizer=regularizers.l2(rf))(l1)

    model = keras.models.Model(inputs=inputs, outputs=output, name='Ej3_TP2')

    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.mean_squared_error,
                  metrics=[metrics.CategoricalAccuracy(name="Acc")])

    model.summary()

    hist = model.fit(x_train,
                     y_train,
                     epochs=epochs,
                     validation_data=(x_val, y_val),
                     batch_size=batch_size,
                     verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '2_EJ3_TP2')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)


def ejercicio_2_EJ4TP2():
    lr = 1e-4
    rf = 1e-3
    batch_size = 128
    epochs = 100
    description = "lr={}_rf={}_bs={}".format(lr, rf, batch_size)

    # Cargo los datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # separo el training en train y validation manteniendo la distribucion (y mezclando)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      stratify=y_train)

    # Hago el flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    x_val = x_val.reshape(len(x_val), x_val[0].size).astype(np.float)

    # Paso los labels a one-hot representation
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

    # Normalizacion
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train = x_train - media
    x_train /= sigma
    x_test = x_test - media
    x_test /= sigma
    x_val = x_val - media
    x_val /= sigma

    # Arquitectura de la red segun el ej4 TP2
    inputs = layers.Input(shape=x_train.shape[1], name="Input")

    l1 = layers.Dense(
        100,
        name='Hidden',
        activation=activations.sigmoid,
        kernel_regularizer=regularizers.l2(rf),
    )(inputs)

    output = layers.Dense(10,
                          name='Output',
                          activation=activations.linear,
                          kernel_regularizer=regularizers.l2(rf))(l1)

    model = keras.models.Model(inputs=inputs, outputs=output, name='Ej4_TP2')

    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.CategoricalAccuracy(name="Acc")])

    model.summary()

    hist = model.fit(x_train,
                     y_train,
                     epochs=epochs,
                     validation_data=(x_val, y_val),
                     batch_size=batch_size,
                     verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)


def ejercicio_2_XOR_A():
    lr = 1e-3
    rf = 0
    batch_size = None
    epochs = 10000
    description = "lr={}_rf={}_bs={}".format(lr, rf, batch_size)

    # Datos que no son datos
    x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y_train = np.array([[1], [-1], [-1], [1]])

    # Arquitectura de la red
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    layer_1 = layers.Dense(2, name='Hidden_1',
                           activation=activations.tanh)(inputs)

    outputs = layers.Dense(1, name='Output',
                           activation=activations.tanh)(layer_1)

    model = keras.models.Model(inputs=inputs,
                               outputs=outputs,
                               name='XOR_Arquitectura_1')

    # Defino accuracy para el problema de XOR
    def my_acc(y_true, y_pred):
        acc = tf.reduce_mean(
            tf.cast(tf.less_equal(tf.abs(y_true - y_pred), 0.1), tf.float32))
        return acc

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=losses.MSE,
                  metrics=[my_acc])

    model.summary()

    hist = model.fit(x_train, y_train, epochs=epochs, verbose=2)

    # Guardo los datos
    data_folder = os.path.join('Datos', '2_XOR_A')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)


def ejercicio_2_XOR_B():
    lr = 1e-3
    rf = 0
    batch_size = None
    epochs = 10000
    description = "lr={}_rf={}_bs={}".format(lr, rf, batch_size)

    # Datos que no son datos
    x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y_train = np.array([[1], [-1], [-1], [1]])

    # Arquitectura de la red
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    layer_1 = layers.Dense(1, name='Hidden_1',
                           activation=activations.tanh)(inputs)

    concat = layers.Concatenate()([inputs, layer_1])

    outputs = layers.Dense(1, name='Output',
                           activation=activations.tanh)(concat)

    model = keras.models.Model(inputs=inputs,
                               outputs=outputs,
                               name='XOR_Arquitectura_2')

    # Defino accuracy para el problema de XOR
    def my_acc(y_true, y_pred):
        acc = tf.reduce_mean(
            tf.cast(tf.less_equal(tf.abs(y_true - y_pred), 0.1), tf.float32))
        return acc

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=losses.MSE,
                  metrics=[my_acc])

    model.summary()

    hist = model.fit(x_train, y_train, epochs=epochs, verbose=2)

    # Guardo los datos
    data_folder = os.path.join('Datos', '2_XOR_B')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)

    # Guardo las imagenes
    img_folder = os.path.join('Figuras', '2_XOR_B')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)


#--------------------------------------
#           Ejercicio 3
#--------------------------------------


def ejercicio_3_BN():
    lr = 1e-4
    rf = 1e-3
    batch_size = 128
    epochs = 100
    nn = 25
    description = "lr={}_rf={}_bs={}_nn={}".format(lr, rf, batch_size, nn)

    # importo los datos
    dim = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

    # Muchos datos de test, prefiero dividirlo en proporciones distintas
    x_train, y_train = np.hstack((x_train, x_test)), np.hstack(
        (y_train, y_test))
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        stratify=y_train)
    # Ahora separa entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      stratify=y_train)

    # Esto no hace falta, era para pasar a texto la reseña
    indexes = imdb.get_word_index()
    r_indexes = dict([(val, key) for key, val in indexes.items()])

    # Funcion que Vectoriza datos teniendo en cuenta repeticiones
    def vectorizeWCounts(x, dim):
        res = np.zeros((len(x), dim))
        for i, sequence in enumerate(x):
            values, counts = np.unique(sequence, return_counts=True)
            res[i, values] = counts
        return res

    # Vectorizo los datos
    x_train = vectorizeWCounts(x_train, dim)
    x_test = vectorizeWCounts(x_test, dim)
    x_val = vectorizeWCounts(x_val, dim)
    y_train = y_train.astype(np.float)
    y_test = y_test.astype(np.float)
    y_val = y_val.astype(np.float)

    # Arquitectura con dropout
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    l1 = layers.Dense(nn, activation=activations.relu, name="Hidden_1")(inputs)

    bn_1 = layers.BatchNormalization()(l1)

    l2 = layers.Dense(nn, activation=activations.relu, name="Hidden_2")(bn_1)

    bn_2 = layers.BatchNormalization()(l2)

    outputs = layers.Dense(1, activation=activations.linear,
                           name="Output")(bn_2)

    model = keras.models.Model(inputs=inputs,
                               outputs=outputs,
                               name="Ejercicio_3_BN")

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.BinaryCrossentropy(from_logits=True,
                                                 name='loss'),
                  metrics=[metrics.BinaryAccuracy(name='B_Acc')])

    model.summary()

    # Entreno
    hist = model.fit(x_train,
                     y_train,
                     validation_data=(x_val, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '3_BN')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)

    # Guardo las imagenes
    img_folder = os.path.join('Figuras', '3_BN')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)


def ejercicio_3_Dropout():
    lr = 1e-4
    rf = 1e-3
    batch_size = 128
    epochs = 100
    nn = 25
    drop_arg = 0.5
    description = "lr={}_rf={}_bs={}_nn={}_do={}".format(
        lr, rf, batch_size, nn, drop_arg)

    # importo los datos
    dim = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

    # Muchos datos de test, prefiero dividirlo en proporciones distintas
    x_train, y_train = np.hstack((x_train, x_test)), np.hstack(
        (y_train, y_test))
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        stratify=y_train)
    # Ahora separa entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      stratify=y_train)

    # Esto no hace falta, era para pasar a texto la reseña
    indexes = imdb.get_word_index()
    r_indexes = dict([(val, key) for key, val in indexes.items()])

    # Funcion que Vectoriza datos teniendo en cuenta repeticiones
    def vectorizeWCounts(x, dim):
        res = np.zeros((len(x), dim))
        for i, sequence in enumerate(x):
            values, counts = np.unique(sequence, return_counts=True)
            res[i, values] = counts
        return res

    # Vectorizo los datos
    x_train = vectorizeWCounts(x_train, dim)
    x_test = vectorizeWCounts(x_test, dim)
    x_val = vectorizeWCounts(x_val, dim)
    y_train = y_train.astype(np.float)
    y_test = y_test.astype(np.float)
    y_val = y_val.astype(np.float)

    # Arquitectura con dropout
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    l1 = layers.Dense(nn, activation=activations.relu, name="Hidden_1")(inputs)

    drop_1 = layers.Dropout(drop_arg)(l1)

    l2 = layers.Dense(nn, activation=activations.relu, name="Hidden_2")(drop_1)

    drop_2 = layers.Dropout(drop_arg)(l2)

    outputs = layers.Dense(1, activation=activations.linear,
                           name="Output")(drop_2)

    model = keras.models.Model(inputs=inputs,
                               outputs=outputs,
                               name="Ejercicio_3_Dropout")

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.BinaryCrossentropy(from_logits=True,
                                                 name='loss'),
                  metrics=[metrics.BinaryAccuracy(name='B_Acc')])

    model.summary()

    # Entreno
    hist = model.fit(x_train,
                     y_train,
                     validation_data=(x_val, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '3_Dropout')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)


def ejercicio_3_L2():
    lr = 1e-3
    rf = 1e-1
    batch_size = 128
    epochs = 100
    nn = 25
    drop_arg = 0.5
    description = "lr={}_rf={}_bs={}_nn={}_do={}".format(
        lr, rf, batch_size, nn, drop_arg)

    # importo los datos
    dim = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

    # Muchos datos de test, prefiero dividirlo en proporciones distintas
    x_train, y_train = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    # Ahora separa entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.25,
                                                    stratify=y_train)

    # Esto no hace falta, era para pasar a texto la reseña
    indexes = imdb.get_word_index()
    r_indexes = dict([(val, key) for key, val in indexes.items()])


    # Funcion que Vectoriza datos teniendo en cuenta repeticiones
    def vectorizeWCounts(x, dim):
        res = np.zeros((len(x), dim))
        for i, sequence in enumerate(x):
            values, counts = np.unique(sequence, return_counts=True)
            res[i, values] = counts
        return res


    # Vectorizo los datos
    x_train = vectorizeWCounts(x_train, dim)
    x_test = vectorizeWCounts(x_test, dim)
    x_val = vectorizeWCounts(x_val, dim)
    y_train = y_train.astype(np.float)
    y_test = y_test.astype(np.float)
    y_val = y_val.astype(np.float)

    # Arquitectura con regularizadores
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    l1 = layers.Dense(nn,
                    activation=activations.relu,
                    kernel_regularizer=regularizers.l2(rf),
                    name="Hidden_1")(inputs)

    l2 = layers.Dense(nn,
                    activation=activations.relu,
                    kernel_regularizer=regularizers.l2(rf),
                    name="Hidden_2")(l1)

    outputs = layers.Dense(1, activation=activations.linear, name="Output")(l2)

    model = keras.models.Model(inputs=inputs,
                            outputs=outputs,
                            name="Ejercicio_3_Regularizadores")

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
                    verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '3_Regularizadores')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

    # Guardo las imagenes
    img_folder = os.path.join('Figuras', '3_Regularizadores')
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

#--------------------------------------
#           Ejercicio 4
#--------------------------------------

def ejercicio_4_Embedding():
    lr = 1e-3
    epochs = 100
    batch_size = 256
    embedding_dim = 75
    nn = 25
    description = "lr={}_bs={}_nn={}".format(
        lr, batch_size, nn)

    # importo los datos
    dim = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

    # Muchos datos de test, prefiero dividirlo en proporciones distintas
    x_train, y_train = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
    # Hacemos el padding ahora que estan todos los datos juntos
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        padding='post',
                                                        dtype=np.float)
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        stratify=y_train)
    # Ahora separo entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.25,
                                                    stratify=y_train)

    # Arquitectura con dropout
    model = keras.models.Sequential(name='Ejercicio_4_Embedding')

    model.add(layers.Embedding(dim, embedding_dim, input_length=x_train.shape[1]))
    model.add(layers.Flatten())

    model.add(layers.Dense(nn, activation=activations.relu, name="Hidden_1"))
    model.add(layers.Dense(nn, activation=activations.relu, name="Hidden_2"))
    model.add(layers.Dense(1, activation=activations.linear, name="Output"))

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
                    verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '4_Embedding')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

def ejercicio_4_Conv():
    lr = 1e-3
    rf = 1e-3
    epochs = 100
    batch_size = 256
    embedding_dim = 75
    drop_arg = 0.2
    nn = 25
    description = "lr={}_bs={}_nn={}".format(
        lr, batch_size, nn)
    
    # importo los datos
    dim = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dim)

    # Muchos datos de test, prefiero dividirlo en proporciones distintas
    x_train, y_train = np.hstack((x_train, x_test)), np.hstack((y_train, y_test))
    # Hacemos el padding ahora que estan todos los datos juntos
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                        padding='post',
                                                        dtype=np.float)
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.2,
                                                        stratify=y_train)
    # Ahora separo entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.25,
                                                    stratify=y_train)

    # Arquitecura con Convolucionales
    model = keras.models.Sequential(name='Ejercicio_4_Conv')

    model.add(layers.Embedding(dim, embedding_dim, input_length=x_train.shape[1]))

    model.add(layers.Dropout(drop_arg))
    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation=activations.relu,
            kernel_regularizer=regularizers.l2(rf),
        ))
    model.add(layers.MaxPooling1D())

    model.add(layers.Dropout(drop_arg))
    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            activation=activations.relu,
            kernel_regularizer=regularizers.l2(rf),
        ))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())

    model.add(layers.Dropout(drop_arg))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation=activations.linear, name="Output"))

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
                    verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '4_Conv')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

#--------------------------------------
#           Ejercicio 5
#--------------------------------------

def ejercicio_5():
    lr = 1e-3
    rf = 1e-3
    epochs = 200
    batch_size = 1024
    description = "lr={}_bs={}".format(lr, batch_size)

    # Datos
    nData = 1000000
    x = np.linspace(0, 1, nData).reshape((nData, 1))
    y = 4 * x * (1 - x)

    # Separo los datos de test
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1)
    # Ahora separo entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1 / 9)

    # Arquitectura de la red
    inputs = layers.Input(shape=(x_train.shape[1], ), name="Input")

    layer_1 = layers.Dense(5,
                        activation=activations.tanh,
                        kernel_regularizer=regularizers.l2(rf),
                        name='Hidden')(inputs)

    concat = layers.Concatenate()([inputs, layer_1])

    outputs = layers.Dense(1,
                        activation=activations.linear,
                        kernel_regularizer=regularizers.l2(rf),
                        name='Output')(concat)

    model = keras.models.Model(inputs=inputs, outputs=outputs, name='Ejercicio_5')

    model.compile(optimizer=optimizers.SGD(learning_rate=lr),
                loss=losses.MeanSquaredError(name='loss'),
                metrics=[metrics.MeanSquaredError(name='acc_MSE')])

    model.summary()

    # Entreno
    hist = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_Acc = model.evaluate(x_test, y_test)

    # Guardo los datos
    data_folder = os.path.join('Datos', '5')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

#--------------------------------------
#           Ejercicio 6
#--------------------------------------

# Funcion auxiliar
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

def ejercicio_6(nn,correccion=False, masCapas=False):
    lr = 1e-3
    rf = 1e-3
    epochs = 500
    batch_size = 32
    description = "lr={}_bs={}".format(lr, batch_size)
    # Cargo los datos
    # Probar esto desde el cluster. Edit: Parece que funciona
    path_folder = os.path.join("/", "share", "apps", "DeepLearning", "Datos")
    file = "pima-indians-diabetes.csv"
    path_file = os.path.join(path_folder, file)
    path_file = '/run/user/1000/gvfs/sftp:host=10.73.25.223,user=facundo.cabrera/share/apps/DeepLearning/Datos/pima-indians-diabetes.csv'

    try:
        data = np.loadtxt(path_file, delimiter=',')
    except :
        print("Ej 6 - No se encontro el archivo pima-indians-diabetes.csv")
        return


    x = data[:, :-1]
    y = data[:, -1].reshape((data.shape[0], 1))

    if correccion:
        # Correcion para los datos que tienen muchos ceros
        # Primero saco las columnas 3 y 4 porque tienen demasiados ceros
        # En principio se puede reemplazar por la media, pero me parecen muchos datos
        # a modificar, prefiero probar sacando esos datos.
        x = np.delete(x, [3,4], 1)
        # Las que ahora soon las columnas 1, 2 y 3 tambien tienen ceros , pero son
        # menos, asi que en vez de eliminar esos datos lo que hago es reemplazar por
        # la media del resto de datos
        means = np.copy(x)
        means[:,1:4][means[:,1:4] == 0] = np.nan    # Reamplazo los 0 por nan
        means = np.nanmean(means[:,1:4], axis=0)    # Calculo mean sin los nan
        # Reemplazo los ceros por la meadia calculada
        x[:,1][x[:,1] == 0] = means[0]
        x[:,2][x[:,2] == 0] = means[1]
        x[:,3][x[:,3] == 0] = means[2]

    # Arquitectura de la red
    model = keras.models.Sequential(name='Ejercicio_6')

    model.add(
        layers.Dense(nn,
                    input_shape=(x.shape[1], ),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(rf)))

    model.add(
        layers.Dense(nn,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(rf)))
    
    if masCapas:
        model.add(
        layers.Dense(nn,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(rf)))
        model.add(
        layers.Dense(nn,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(rf)))


    model.add(layers.Dense(1, activation=activations.linear, name="Output"))

    model.compile(optimizer=optimizers.SGD(learning_rate=lr),
                loss=losses.BinaryCrossentropy(from_logits=True, name='loss'),
                metrics=[metrics.BinaryAccuracy(name='Acc')])

    model.summary()

    # Guardo los pesos para cargarlos y "ressetear" el modelo en cada fold
    data_folder = os.path.join('Datos', '6_CORRECION')
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

    results = toDictionary(acc_test, acc_train, loss_test, loss_train)
    np.save(os.path.join(data_folder, '{}.npy'.format(description)), results)

#--------------------------------------
#           Ejercicio 7
#--------------------------------------

#--------------------------------------
#           Ejercicio 8
#--------------------------------------

#--------------------------------------
#           Ejercicio 9
#--------------------------------------

#--------------------------------------
#           Ejercicio 10
#--------------------------------------

#--------------------------------------
#   Funciones para hacer los graficos
#--------------------------------------


def graficos_2_EJ3TP2():
    path_data = os.path.join(
        "Informe", "Datos", "2_EJ3_TP2",
        "lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model = os.path.join(
        "Informe", "Datos", "2_EJ3_TP2",
        "lr=0.0001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_EJ3_TP2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['Acc'], label="Acc. Training")
    plt.plot(data['val_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_2_EJ4TP2():
    path_data = os.path.join(
        "Informe", "Datos", "2_EJ4_TP2",
        "lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.npy")
    path_model = os.path.join(
        "Informe", "Datos", "2_EJ4_TP2",
        "lr=0.001_rf=0.001_do=1_epochs=300_bs=128_nn=10.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_EJ4_TP2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['Acc'], label="Acc. Training")
    plt.plot(data['val_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_2_XOR():
    path_data_a = os.path.join(
        "Informe", "Datos", "2_XOR_A",
        "lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.npy")
    path_model_a = os.path.join(
        "Informe", "Datos", "2_XOR_A",
        "lr=0.001_rf=0_do=1_epochs=10000_bs=None_nn=10.h5")

    path_data_b = os.path.join(
        "Informe", "Datos", "2_XOR_B",
        "lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.npy")
    path_model_b = os.path.join(
        "Informe", "Datos", "2_XOR_B",
        "lr=1.0e-03_rf=0.0e+00_do=0_epochs=10000_bs=None_nn=10_ed=100.h5")

    data_a = np.load(path_data_a, allow_pickle=True).item()
    # model_a = keras.models.load_model(path_model_a)

    data_b = np.load(path_data_b, allow_pickle=True).item()
    # model_b = keras.models.load_model(path_model_b)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '2_XOR')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data_a['loss'], label="1° Arq.")
    plt.plot(data_b['loss'], label="2° Arq.")
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data_a['my_acc'], label="1° Arq.")
    plt.plot(data_b['my_acc'], label="2° Arq.")
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_3_BN():
    path_data = os.path.join(
        "Informe", "Datos", "3_BN",
        "lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_BN",
        "lr=0.0001_rf=0_do=1_epochs=200_bs=128_nn=25.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_BN')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_3_Dropout():
    path_data = os.path.join(
        "Informe", "Datos", "3_Dropout",
        "lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_Dropout",
        "lr=0.0001_rf=0.0_do=0.75_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_Dropout')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_3_L2():
    path_data = os.path.join(
        "Informe", "Datos", "3_L2",
        "lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "3_L2",
        "lr=0.0001_rf=0.1_do=0_epochs=200_bs=64_nn=25_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_L2')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_4_Embedding():
    path_data = os.path.join(
        "Informe", "Datos", "4_Embedding",
        "lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.npy")
    path_model = os.path.join(
        "Informe", "Datos", "4_Embedding",
        "lr=1.0e-05_rf=0_do=0_epochs=200_bs=256_nn=25_ed=75.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '4_Embedding')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'][:100], label="Acc. Training")
    plt.plot(data['val_B_Acc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_4_Conv():
    path_data = os.path.join(
        "Informe", "Datos", "4_Conv",
        "lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.npy")
    path_model = os.path.join(
        "Informe", "Datos", "4_Conv",
        "lr=1.0e-05_rf=0_do=0.2_epochs=200_bs=256_nn=10_ed=50.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '4_Conv')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'], label="Acc. Training")
    plt.plot(data['val_B_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def mapeoLogistico(x):
    return 4 * x * (1 - x)


def graficos_5():
    path_data = os.path.join(
        "Informe", "Datos", "5",
        "lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "5",
        "lr=1.0e-03_rf=0.0_do=0_epochs=200_bs=256_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '5')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['B_Acc'], label="Acc. Training")
    plt.plot(data['val_B_Acc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_6():
    path_data = os.path.join(
        "Informe", "Datos", "6",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=20_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_6_40N():
    path_data = os.path.join(
        "Informe", "Datos", "6",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=40_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6_40')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_6_CCorreccion():
    path_data = os.path.join(
        "Informe", "Datos", "6_Correccion",
        "lr=1.0e-03_rf=1.0e-01_do=0_epochs=500_bs=32_nn=40_ed=100.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '6_Correccion')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    xx = np.arange(len(data['train_min']))
    plt.fill_between(xx, data['train_min'], data['train_max'], alpha=0.45)
    plt.plot(data['train_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    # plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.fill_between(xx, data['test_min'], data['test_max'], alpha=0.45)
    plt.plot(data['test_mean'])
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Test Accuracy", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc_Test.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.fill_between(xx, data['ltrain_min'], data['ltrain_max'], alpha=0.45)
    plt.plot(data['ltrain_mean'], label="Loss Training")
    plt.fill_between(xx, data['ltest_min'], data['ltest_max'], alpha=0.45)
    plt.plot(data['ltest_mean'], label="Loss Test")
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_7():
    path_data = os.path.join(
        "Informe", "Datos", "7",
        "lr=1.0e-05_rf=0.0e+00_do=0_epochs=500_bs=256_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "7",
        "lr=1.0e-05_rf=0.0e+00_do=0_epochs=500_bs=256_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '7')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['MSE'], label="Loss Training")
    plt.plot(data['val_MSE'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_7_v2():
    path_data = os.path.join(
        "Informe", "Datos", "7_Internet",
        "MSE_lr=1.0e-05_rf=0.0e+00_do=0_epochs=1000_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "7_Internet",
        "MSE_lr=1.0e-05_rf=0.0e+00_do=0_epochs=1000_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '7_Internet')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['MSE'], label="Loss Training")
    plt.plot(data['val_MSE'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_8_Densa():
    path_data = os.path.join(
        "Informe", "Datos", "8_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "8_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '8_Dense')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_8_Conv():
    path_data = os.path.join(
        "Informe", "Datos", "8_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "8_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '8_Conv')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_9_Densa():
    path_data = os.path.join(
        "Informe", "Datos", "9_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "9_Dense",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '9_Dense')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_9_Conv():
    path_data = os.path.join(
        "Informe", "Datos", "9_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.npy")
    path_model = os.path.join(
        "Informe", "Datos", "9_Conv",
        "lr=1.0e-04_rf=1.0e-03_do=0.4_epochs=200_bs=512_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '9_Conv')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:100], label="Loss Training")
    plt.plot(data['val_loss'][:100], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'][:100], label="Acc. Training")
    plt.plot(data['val_CAcc'][:100], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_AlexNet_Cifar10():
    path_data = os.path.join(
        "Informe", "Datos", "10_AlexNet_cifar10",
        "lr=1.0e-03_rf=1.0e-03_do=0_epochs=100_bs=64_nn=10_ed=100.npy")
    # path_model  = os.path.join("Informe","Datos","10_AlexNet_cifar10","lr=1.0e-03_rf=1.0e-03_do=0_epochs=100_bs=64_nn=10_ed=100.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_AlexNet_cifar10')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_AlexNet_Cifar100():
    path_data = os.path.join("Informe", "Datos", "10_AlexNet_cifar100",
                             "1.0e-03_3.0e-04_100_64_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_AlexNet_cifar10","1.0e-03_3.0e-04_100_64_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_AlexNet_cifar100')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_VGG_Cifar10():
    path_data = os.path.join("Informe", "Datos", "10_VGG16_cifar10",
                             "1.0e-03_6.0e-04_50_256_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_VGG16_cifar10","1.0e-03_6.0e-04_50_256_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_VGG16_cifar10')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def graficos_10_VGG_Cifar100():
    path_data = os.path.join("Informe", "Datos", "10_VGG16_cifar100",
                             "1.0e-03_6.0e-04_100_256_SINDROP.npy")
    # path_model  = os.path.join("Informe","Datos","10_VGG16_cifar10","1.0e-03_6.0e-04_100_256_SINDROP.h5")

    data = np.load(path_data, allow_pickle=True).item()
    # model = keras.models.load_model(path_model)

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '10_VGG16_cifar100')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'], label="Loss Training")
    plt.plot(data['val_loss'], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(data['CAcc'], label="Acc. Training")
    plt.plot(data['val_CAcc'], label="Acc. Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    # ejercicio_1()

    # ejercicio_2_EJ3TP2()
    # ejercicio_2_EJ4TP2()
    # ejercicio_2_XOR_A()
    # ejercicio_2_XOR_B()

    # graficos_2_EJ3TP2()
    # graficos_2_EJ4TP2()
    # graficos_2_XOR()

    # ejercicio_3_BN()
    # ejercicio_3_Dropout()
    # ejercicio_3_L2()

    # graficos_3_BN()
    # graficos_3_Dropout()
    # graficos_3_L2()

    # ejercicio_4_Embedding()
    # ejercicio_4_Conv()

    # graficos_4_Embedding()
    # graficos_4_Conv()

    # ejercicio_5()
    # graficos_5()

    # ejercicio_6(20)
    # ejercicio_6(40)
    # ejercicio_6(20,correccion=True)
    # ejercicio_6(20,masCapas=True)

    # graficos_6()
    # graficos_6_40N()
    # graficos_6_CCorreccion()

    # graficos_7()

    # graficos_7_v2()

    # graficos_8_Densa()

    # graficos_8_Conv()

    # graficos_9_Densa()

    # graficos_9_Conv()

    # graficos_10_AlexNet_Cifar10()

    # graficos_10_AlexNet_Cifar100()

    # graficos_10_VGG_Cifar10()

    # graficos_10_VGG_Cifar100()
