#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 16-09-2020
File: ej_05.py
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
snn.set(font_scale = 1.2)





# Cargo los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten de los datos
x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)

media = x_train.mean(axis=0)
sigma = x_train.std(axis=0)

x_train = x_train - media
x_train /= sigma
# x_train /= 255
# x_train = np.hstack((np.ones((len(x_train),1)), x_train))              # Pongo 1 para el bias
y_train = y_train.reshape(y_train.size)

x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
x_test = x_test - media
x_test /= sigma
# x_test /= 255
# x_test = np.hstack((np.ones((len(x_test),1)), x_test))        # Pongo 1 para el bias
y_test = y_test.reshape(y_test.size)




def yyZeros(y_true, shape_):
    zeros = np.zeros(shape = shape_)
    zeros[ np.arange(y_true.shape[0]), y_true] = 1
    return zeros

def Scores(x_,W_):
    return np.dot(x_,W_)

def predict(scores):
    return np.argmax(scores, axis=1)  

def accuracy(scores, y_true):
    y_predict = predict(scores)
    return (y_predict == y_true).mean()*100

def mse(scores, y_true):
    yy = yyZeros(y_true, scores.shape)
    return ((scores-yy)**2).sum(axis=1).mean()

def grad_MSE(scores, y_true):
    yy = yyZeros(y_true, scores.shape)
    return 2*(scores-yy)/len(y_true)

def sigmoid(x_):
    return 1/(1 + np.exp(-x_))

def grad_sigmoid(x_):
    sigma = sigmoid(x_)
    return (1-sigma)*sigma

def lossSoftmax(scores_, y_true):
    scores = np.copy(scores_)

    scores = scores - scores.max(axis=1)[:,np.newaxis]

    y_idx = np.arange(y_true.size)
    y_win = scores[y_idx, y_true]

    exp = np.exp(scores)

    sumatoria = exp.sum(axis=1)

    log_softmax = np.log(sumatoria) - y_win

    return log_softmax.mean()

def grad_Softmax(scores, y_true):

    scores = scores - scores.max(axis=1)[:,np.newaxis]

    exp = np.exp(scores)

    sumatoria = exp.sum(axis=1)
    softmax_fun = (1.0/sumatoria)[:,np.newaxis] * exp
    softmax_fun[np.arange(y_true.size), y_true] -= 1

    return softmax_fun / len(y_true)

def lineal(x_):
    return x_

def gradLineal(x_):
    return 1

def Relu(x_):
    return np.maximum(0,x_)

def gradRelu(x_):
    return np.heaviside(x_, 0)



def fit(x, y, x_test, y_test, size_bacht=50, lr=1e-3, landa=1e-2, epochs=200, w=1e-2, loss_f=mse, grad_f=grad_MSE,
                    act_1=sigmoid, grad_1=grad_sigmoid, act_2=lineal, grad_2=gradLineal, lPlot=True, ej=0):
    
    n, dim = x.shape
    clases = 10
    n_neuronas = 100

    idx = np.arange(n)

    loss = np.array([])
    acc  = np.array([])
    t_acc = np.array([])

    W1 = np.random.uniform(-1,1, size=(dim+1, n_neuronas)) * w
    W2 = np.random.uniform(-1,1, size=(n_neuronas+1, clases)) * w

    x_t = np.hstack((np.ones((len(x_test),1)), x_test))
    x = np.hstack((np.ones((len(x),1)), x))

    n_bacht = int(len(x)/size_bacht)

    if(lPlot):
        fig    = plt.figure(figsize=(11,4))
        axloss = plt.subplot(131)
        axacc  = plt.subplot(132)
        axTacc = plt.subplot(133)


    for e in range(epochs):
        log_loss = 0
        log_acc  = 0

        np.random.shuffle(idx)  # Mezclo los indices

        for i in range(n_bacht):

            bIdx = idx[size_bacht*i: size_bacht*(i+1)]

            x_bacht = x[bIdx]
            y_bacht = y[bIdx]

            # Capa 1
            Y1 = np.dot(x_bacht, W1)
            S1 = act_1( Y1 )

            # Capa 2
            S1 = np.hstack((np.ones((len(S1),1)), S1))
            Y2 = np.dot(S1, W2)
            S2 = act_2( Y2 )

            # Regularizacion
            reg1 = np.sum(W1 * W1)
            reg2 = np.sum(W2 * W2)
            reg  = 0.5 * landa * (reg1 + reg2)

            log_loss += loss_f(S2, y_bacht) + reg
            log_acc  += accuracy(S2, y_bacht)


            # Ahora arranca el backpropagation
            grad = grad_f(S2, y_bacht)  # Este gradiente ya tiene hecho el promedio
            grad2 = grad_2( Y2 )

            grad = grad * grad2

            # Capa 2
            dW2 = np.dot(S1.T, grad)    # El grad ya tiene el promedio en bachts

            grad = np.dot(grad, W2.T)
            grad = grad[:, 1:]  # saco la colunmas correspondiente al bias

            # Capa 1
            grad1 = grad_1( Y1 )

            grad = grad * grad1

            dW1 = np.dot(x_bacht.T, grad) # El grad ya tiene el promedio en bachts

            # Actualizo las W
            W1 -= (lr * (dW1 + landa*W1))
            W2 -= (lr * (dW2 + landa*W2))
        

        # Forward para el test
        S1_t = act_1( np.dot(x_t , W1) )
        S1_t = np.hstack((np.ones((len(S1_t),1)), S1_t))
        S2_t = act_2( np.dot(S1_t, W2) )

        t_acc = np.append(t_acc, accuracy(S2_t, y_test))

        loss = np.append(loss, log_loss/n_bacht)
        acc  = np.append(acc , log_acc/n_bacht)

        if (e % 10 == 0):
            # print(e,"/",epochs, "\tloss:",log_loss/n_bacht, "\tAccurracy: ",log_acc/n_bacht, "\ttest: ", accuracy(S2_t, y_test) )
            print("{}/{}\tloss: {:.2f}\tAccurracy: {:.2f}\ttest: {:.2f}".format(
                e, epochs, log_loss/n_bacht, log_acc/n_bacht, accuracy(S2_t, y_test)))
        
        if(lPlot):

            fig.clf()
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
    

    if(lPlot):
        plt.close('all')
    plt.ion()

    plt.figure()
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.plot(np.arange(epochs), loss)
    if(ej != 0):
        plt.savefig('Informe/345/Loss_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')

    plt.figure()
    plt.ylabel("Accurracy Training")
    plt.xlabel("Epoca")
    plt.plot(np.arange(epochs), acc)
    if(ej != 0):
        plt.savefig('Informe/345/Acc_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')

    plt.figure()
    plt.ylabel("Accurracy Test")
    plt.xlabel("Epoca")
    plt.plot(np.arange(epochs), t_acc)
    if(ej != 0):
        plt.savefig('Informe/345/Acc_t_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')
        np.savez("Informe/345/datos_ej_{}.npz".format(ej), Loss=loss, Acc=acc, Acc_t=t_acc)
    
    plt.pause(5)
    plt.close('all')



# Para el ejercicio 3
np.random.seed(10)

fit(x_train, y_train, x_test, y_test,
        epochs=200, lr=1e-3, landa=1e-2, w=1e-2,
        loss_f=mse, grad_f=grad_MSE,
        act_1=sigmoid, grad_1=grad_sigmoid, act_2=lineal, grad_2=gradLineal,
        lPlot=False, ej=3)


# Para el ejercicio 4
np.random.seed(14)

fit(x_train, y_train, x_test, y_test,
        epochs=200, lr=1e-3, landa=1e-2, w=1e-2,
        loss_f=lossSoftmax, grad_f=grad_Softmax,
        act_1=sigmoid, grad_1=grad_sigmoid, act_2=lineal, grad_2=gradLineal,
        lPlot=False, ej=4)



# Para el ejercicio 5
np.random.seed(14)

fit(x_train, y_train, x_test, y_test,
            epochs=200, lr=5e-3, landa=1e-2, w=1e-2,
            loss_f=lossSoftmax, grad_f=grad_Softmax,
            act_1=Relu, grad_1=gradRelu, act_2=sigmoid, grad_2=grad_sigmoid,
            lPlot=False, ej=5)        # Este se dio bien, pero se puede mejorar