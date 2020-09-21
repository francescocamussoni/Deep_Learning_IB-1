#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 16-09-2020
File: ej_04.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from keras.datasets import cifar10
from keras.datasets import mnist

import seaborn as snn
snn.set(font_scale = 1.1)



# np.random.seed(14)



# Cargo los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten de los datos
x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)

media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
# x_train = np.hstack((np.ones((len(x_train),1)), x_train))              # Pongo 1 para el bias
y_train = y_train.reshape(y_train.size)

x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
x_test = x_test - media
x_test /= sigma
# x_test = np.hstack((np.ones((len(x_test),1)), x_test))        # Pongo 1 para el bias
y_test = y_test.reshape(y_test.size)

# yy = yyZeros(y_train)


# Constantes y cosas
"""
n = 3       # Numero de ejemplos
dim = 8     # Dimension
clases = 10 # Numero de clases
"""
n, dim = x_train.shape
# n = len(x_train)
# dim = 
clases = 10
n_neuronas = 100

idx = np.arange(n)



# esta es fea, le estoy metiendo medio a mano
def yyZeros(y_true):
    zeros = np.zeros(shape = (y_true.shape[0],clases))
    zeros[np.arange(y_true.shape[0]), y_true] = 1
    return zeros


# yy = yyZeros(y_train)

# Inicializo la matriz de pesos
# W1 = np.random.uniform(-1,1, size=(dim+1, n_neuronas)) * 1e-3
# W2 = np.random.uniform(-1,1, size=(n_neuronas+1, clases)) * 1e-3







# x = np.random.randint(-10,10, size=(n, dim) )
# W = np.random.randint(-10,10, size=(dim, clases))
# y = np.random.randint(0,10, size=(1,n))


def Scores(x_,W_):
    return np.dot(x_,W_)

def predict(scores):
    return np.argmax(scores, axis=1)
     

def accuracy(scores, y_true):
    y_predict = predict(scores)
    return (y_predict == y_true).mean()*100

# esta es fea, le estoy metiendo medio a mano
# def yyZeros(y_true):
#     zeros = np.zeros(shape = (n,clases))
#     zeros[np.arange(n), y_true] = 1
#     return zeros

# Metrica MSE

def MSE(scores, y_true):
    # Mejor restar directamente a a una copia de Scores
    yy = yyZeros(y_true)
    return ((scores-yy)**2).sum(axis=1).mean()

def grad_MSE(scores, y_true):
    yy = yyZeros(y_true)
    return 2*(scores-yy)/len(y_true)

# Funcion de costo sigmoide
def sigmoid(x_):
    return 1/(1 + np.exp(-x_))

def grad_sigmoid(x_):
    sigma = sigmoid(x_)
    return (1-sigma)*sigma

# XXX juntar las dos anteriores en una
def lossSoftmax(scores, y_true):
    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    scores = scores - scores.max(axis=1)[:,np.newaxis]

    y_idx = np.arange(y_true.size)
    y_win = scores[y_idx, y_true]

    exp = np.exp(scores)

    sumatoria = exp.sum(axis=1)

    log_softmax = np.log(sumatoria) - y_win

    return log_softmax.mean()

def grad_Softmax(scores, y_true):

    scores = scores - scores.max(axis=1)[:,np.newaxis]

    y_idx = np.arange(y_true.size)
    y_win = scores[y_idx, y_true]

    exp = np.exp(scores)

    sumatoria = exp.sum(axis=1)
    #softmax_fun = exp * (1.0/sumatoria)
    softmax_fun = (1.0/sumatoria)[:,np.newaxis] * exp
    softmax_fun[y_idx, y_true] -= 1

    return softmax_fun / len(y_true)
# log_softmax = np.log(sumatoria) - y_win






def fit(size_bacht=50, lr=1e-5, landa=1e-2, epochs=200, loss_f=MSE, grad_f=grad_MSE):
    loss = np.array([])
    acc  = np.array([])
    t_acc = np.array([])

    W1 = np.random.uniform(-1,1, size=(dim+1, n_neuronas)) * 1e-2
    W2 = np.random.uniform(-1,1, size=(n_neuronas+1, clases)) * 1e-2


    x_t = np.hstack((np.ones((len(x_test),1)), x_test))


    n_bacht = int(len(x_train)/size_bacht)  # Cuantos bacht tengo, pueden que me queden imagenes afuera. Arreglar


    #plt.ion()
    fig    = plt.figure(figsize=(11,4))
    # fig.subplots_adjust(top=0.911,bottom=0.098,left=0.046,right=0.988,hspace=0.2,wspace=0.163)
    #fig.tight_layout()
    #plt.tight_layout()
    #plt.axes([0.025, 0.025, 0.95, 0.95])
    axloss = plt.subplot(131)
    axacc  = plt.subplot(132)
    axTacc = plt.subplot(133)

    # fig,  axloss = plt.subplots()
    # fig2, axacc  = plt.subplots()
    # fig3, axTacc = plt.subplots()
    


    for e in range(epochs):
        log_loss = 0
        log_acc  = 0

        # Mezclo los indices
        np.random.shuffle(idx)

        for i in range(n_bacht):

            bIdx = idx[size_bacht*i: size_bacht*(i+1)]

            x_bacht = x_train[bIdx]
            y_bacht = y_train[bIdx]

            # XXX XXX XXX Tendria que hacer una copia?
            x_bacht = np.hstack((np.ones((len(x_bacht),1)), x_bacht))

            Y1 = np.dot(x_bacht, W1)

            S1 = sigmoid( Y1 )

            S1 = np.hstack((np.ones((len(S1),1)), S1))


            S2 = np.dot(S1, W2)

            # Regularizacion
            reg1 = np.sum(W1 * W1)
            reg2 = np.sum(W2 * W2)
            reg  = 0.5 * landa * (reg1 + reg2)

            log_loss += loss_f(S2, y_bacht) + reg
            log_acc  += accuracy(S2, y_bacht)


            # Ahora arranca el backpropagation
            grad = grad_f(S2, y_bacht)
            # grad = grad_f(S2, y_bacht)

            # Capa 2
            #XXX DEBERIA DIVIDIR POR EL SIZE DEL BACHT?
            dW2 = np.dot(S1.T, grad)

            grad = np.dot(grad, W2.T)
            grad = grad[:, 1:]  # saco la colunmas correspondiente al bias

            # Capa 1
            grad_sig = grad_sigmoid( Y1 )
            #grad_sig = grad_sigmoid( Y1 )

            grad = grad * grad_sig

            #XXX DEBERIA DIVIDIR POR EL SIZE DEL BACHT?
            dW1 = np.dot(x_bacht.T, grad)

            # Actualizo las W
            W1 -= (lr * (dW1 + landa*W1))
            W2 -= (lr * (dW2 + landa*W2))
        
        
        

        # Forward para el test
        S1_t = sigmoid( np.dot(x_t, W1) )
        S1_t = np.hstack((np.ones((len(S1_t),1)), S1_t))
        S2_t = np.dot(S1_t, W2)

        t_acc = np.append(t_acc, accuracy(S2_t, y_test))

        loss = np.append(loss, log_loss/n_bacht)        # Me dijeron que tome el promedio, pero no me gusta
        acc  = np.append(acc , log_acc/n_bacht)

        if (e % 10 == 0):
            # print(e,"/",epochs, "\tloss:",log_loss/n_bacht, "\tAccurracy: ",log_acc/n_bacht, "\ttest: ", accuracy(S2_t, y_test) )
            print("{}/{}\tloss: {:.2f}\tAccurracy: {:.2f}\ttest: {:.2f}".format(
                e, epochs, log_loss/n_bacht, log_acc/n_bacht, accuracy(S2_t, y_test)))
        
        #axloss.cla(), axacc.cla(), axTacc.cla()
        #plt.clf()
        fig.clf()
        # fig.tight_layout()
        axloss = plt.subplot(131)
        axacc  = plt.subplot(132)
        axTacc = plt.subplot(133)
        axloss.set_title("Loss"), axacc.set_title("Accurracy"), axTacc.set_title("Test Accurracy")
        axloss.plot(np.arange(e+1), loss)
        axacc.plot(np.arange(e+1), acc)
        axTacc.plot(np.arange(e+1), t_acc)
        plt.tight_layout()
        plt.pause(0.1)

    
    print("Precision final con los datos de entrenamiento: ", acc[-1])
    print("Precision final con los datos de test: ", t_acc[-1])
    

    plt.close('all')
    plt.ion()

    plt.figure()
    plt.title("Loss")
    plt.plot(np.arange(epochs), loss)

    plt.figure()
    plt.title("Accurracy")
    plt.plot(np.arange(epochs), acc)

    plt.figure()
    plt.title("Test Accurracy")
    plt.plot(np.arange(epochs), t_acc)


# fit(epochs=200, lr=1e-3, landa=1e-2,  loss_f=lossSoftmax, grad_f=grad_Softmax)
fit(epochs=200, lr=1e-3, landa=1e-2,  loss_f=lossSoftmax, grad_f=grad_Softmax)
