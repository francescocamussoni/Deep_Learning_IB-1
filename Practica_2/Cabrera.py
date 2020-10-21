#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 27-09-2020
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
import itertools
from keras.datasets import cifar10

import seaborn as snn
snn.set(font_scale = 1)

#-------------------------------------
# Para los ejercicio 3, 4 y 5
#-------------------------------------


def yyZeros(y_true, shape_):
    zeros = np.zeros(shape = shape_)
    zeros[np.arange(y_true.shape[0]), y_true] = 1
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
    # plt.ion()

    #plt.figure()
    plt.plot(np.arange(epochs), loss)
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    if(ej != 0):
        plt.savefig('Informe/345/Loss_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')
    plt.show()

    # plt.figure()
    plt.plot(np.arange(epochs), acc)
    plt.ylabel("Accurracy Training",fontsize=15)
    plt.xlabel("Epoca",fontsize=15)
    if(ej != 0):
        plt.savefig('Informe/345/Acc_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')
    plt.show()

    # plt.figure()
    plt.plot(np.arange(epochs), t_acc)
    plt.ylabel("Accurracy Test",fontsize=15)
    plt.xlabel("Epoca",fontsize=15)
    if(ej != 0):
        plt.savefig('Informe/345/Acc_t_ej_{}.pdf'.format(ej), format='pdf', bbox_inches='tight')
        np.savez("Informe/345/datos_ej_{}.npz".format(ej), Loss=loss, Acc=acc, Acc_t=t_acc)
    plt.show()
    
    # plt.pause(5)
    # plt.close('all')
    #plt.show()



#-------------------------------------
# Para el ejercicio 3
#-------------------------------------
def ejercicio3():
    print("Ejercicio 3")

    # Cargo los datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)

    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train = x_train - media
    x_train /= sigma
    y_train = y_train.reshape(y_train.size)

    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    x_test = x_test - media
    x_test /= sigma
    y_test = y_test.reshape(y_test.size)

    np.random.seed(10)

    fit(x_train, y_train, x_test, y_test,
            epochs=300, lr=1e-3, landa=1e-2, w=1e-2,
            loss_f=mse, grad_f=grad_MSE,
            act_1=sigmoid, grad_1=grad_sigmoid, act_2=lineal, grad_2=gradLineal,
            lPlot=False, ej=3)


#-------------------------------------
# Para el ejercicio 4
#-------------------------------------
def ejercicio4():
    print("Ejercicio 4")

    # Cargo los datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)

    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train = x_train - media
    x_train /= sigma
    y_train = y_train.reshape(y_train.size)

    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    x_test = x_test - media
    x_test /= sigma
    y_test = y_test.reshape(y_test.size)

    np.random.seed(14)

    fit(x_train, y_train, x_test, y_test,
            epochs=300, lr=1e-3, landa=1e-2, w=1e-2,
            loss_f=lossSoftmax, grad_f=grad_Softmax,
            act_1=sigmoid, grad_1=grad_sigmoid, act_2=lineal, grad_2=gradLineal,
            lPlot=False, ej=4)


#-------------------------------------
# Para el ejercicio 5
#-------------------------------------
def ejercicio5():
    print("Ejercicio 5")
    # Cargo los datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)

    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train = x_train - media
    x_train /= sigma
    y_train = y_train.reshape(y_train.size)

    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    x_test = x_test - media
    x_test /= sigma
    y_test = y_test.reshape(y_test.size)

    np.random.seed(14)

    print("Ejercicio 5 - Relu + Sigmoide con Softmax")
    fit(x_train, y_train, x_test, y_test,
                epochs=300, lr=5e-3, landa=1e-2, w=5e-2,
                loss_f=lossSoftmax, grad_f=grad_Softmax,
                act_1=Relu, grad_1=gradRelu, act_2=sigmoid, grad_2=grad_sigmoid,
                lPlot=False, ej="5_Relu_Sig_Soft")
    
    print("Ejercicio 5 - Relu + Sigmoide con MSE")
    fit(x_train, y_train, x_test, y_test,
                epochs=300, lr=5e-3, landa=1e-2, w=5e-2,
                loss_f=mse, grad_f=grad_MSE,
                act_1=Relu, grad_1=gradRelu, act_2=sigmoid, grad_2=grad_sigmoid,
                lPlot=False, ej="5_Relu_Sig_MSe")
    
    print("Ejercicio 5 - Relu + Lineal con Softmax")
    fit(x_train, y_train, x_test, y_test,
                epochs=300, lr=5e-3, landa=1e-2, w=5e-2,
                loss_f=lossSoftmax, grad_f=grad_Softmax,
                act_1=Relu, grad_1=gradRelu, act_2=lineal, grad_2=gradLineal,
                lPlot=False, ej="5_Relu_Lin_Soft")
    
    print("Ejercicio 5 - Relu + Lineal con MSE")
    fit(x_train, y_train, x_test, y_test,
                epochs=300, lr=5e-3, landa=1e-2, w=5e-2,
                loss_f=mse, grad_f=grad_MSE,
                act_1=Relu, grad_1=gradRelu, act_2=lineal, grad_2=gradLineal,
                lPlot=False, ej="5_Relu_Lin_MSe")


from myModules import models, layers, regularizers, losses, activations, metrics, optimizers

#-------------------------------------
# Para el ejercicio 6
#-------------------------------------
def ejercicio6_1():
    print("Ejercicio 6 - Primera arquitectura")
    np.random.seed(14)

    x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    y_train = np.array([[1],[-1],[-1],[1]])
    y_train = y_train.reshape(y_train.size,1)

    reg1 = regularizers.L2(0)
    reg2 = regularizers.L1(0)

    inputt = layers.Input(x_train.shape[1])

    model = models.Network(inputt)

    # model.add(layers.Dense(units=2,activations.Tanh(),input_dim=x_train.shape[1], regularizer=reg1))
    # model.add(layers.Dense(units=1,activations.Tanh(), regularizer=reg2))

    model.add(layers.Dense(units=2, activation=activations.Tanh(), regu=reg1, w=1))
    model.add(layers.Dense(units=1, activation=activations.Tanh(), regu=reg2, w=1))

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    model.fit(  x=x_train, y=y_train,
                bs= x_train.shape[0], epochs=10000,
                opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR,
                plot=False, print_every=500, ej="6_1")

def ejercicio6_2():
    print("Ejercicio 6 - Segunda arquitectura")
    np.random.seed(14)

    x_train = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    y_train = np.array([[1],[-1],[-1],[1]])
    y_train = y_train.reshape(y_train.size,1)

    reg1 = regularizers.L2(0)
    reg2 = regularizers.L1(0)

    inputt = layers.Input(x_train.shape[1])

    model = models.Network(inputt)

    # model.add(layers.Dense(units=2,activations.Tanh(),input_dim=x_train.shape[1], regularizer=reg1))
    # model.add(layers.Dense(units=1,activations.Tanh(), regularizer=reg2))

    model.add(layers.Dense(units=1, activation=activations.Tanh(), regu=reg1, w=1e-1))
    model.add(layers.Concat(inputt, model.forward, 0))
    model.add(layers.Dense(units=1, activation=activations.Tanh(), regu=reg2, w=1e-1))

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    model.fit(  x=x_train, y=y_train,
                bs= x_train.shape[0], epochs=10000,
                opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR,
                plot=False, print_every=500, ej="6_2")

#-------------------------------------
# Para el ejercicio 7
#-------------------------------------
def ejercicio7(N1,N2):
    print("Ejercicio 7: N1 = {}, N2 = {}".format(N1,N2))

    np.random.seed(69)

    # N1 = 10
    # N2 = 20
    cant_ejemplos = 2**N1

    x_train = np.array([x for x in itertools.product([-1, 1], repeat=N1)])
    y_train = np.prod(x_train, axis=1).reshape(cant_ejemplos, 1)

    # x_test = x_train[-cant_ejemplos//10:]
    # y_test = y_train[-cant_ejemplos//10:]
    # x_train = x_train[:-cant_ejemplos//10]
    # y_train = y_train[:-cant_ejemplos//10]

    reg1 = regularizers.L2(0)
    reg2 = regularizers.L1(0)


    inputt = layers.Input(x_train.shape[1])

    model = models.Network(inputt)

    model.add(layers.Dense(units=N2, activation=activations.Tanh(), regu=reg1, w=1))
    model.add(layers.Dense(units=1,  activation=activations.Tanh(), regu=reg2, w=1))

    # model.fit(x=x_train, y=y_train, bs=x_train.shape[0], epochs=100000, x_test=x_test, y_test=y_test,
            # opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR, plot=False, print_every=500)
    
    model.fit(  x=x_train, y=y_train,
                bs=x_train.shape[0], epochs=100000,
                opt=optimizers.SGD(lr=1e-1), loss=losses.MSE_XOR(), metric=metrics.acc_XOR,
                plot=False, print_every=1000, ej="7_N1={}_N2={}".format(N1,N2))

#-------------------------------------
# Para el ejercicio 8
#-------------------------------------
def ejercicio8():
    print("Ejercicio 8")

    np.random.seed(14)  # Con MSE, landa = 1e-2, lr=1e-3, con std y media, 1e-2 las 2 1ras w


    # Cargo los datos de cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Hago el flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    y_train = y_train.reshape(y_train.size)
    y_test = y_test.reshape(y_test.size)

    # Resto la media y divido por sigma
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train  = x_train - media
    x_train /= sigma    # x_train /= 255
    x_test   = x_test  - media
    x_test  /= sigma

    # n, dim = x_train.shape  # Numero de ejemplos de training y dimension del problema


    reg1 = regularizers.L2(1e-2)
    reg2 = regularizers.L2(1e-2)
    reg3 = regularizers.L2(1e-2)


    inputt = layers.Input(x_train.shape[1])

    model = models.Network(inputt)

    # Capa Oculta 1
    model.add(layers.Dense(units=100, activation=activations.Relu(), regu=reg1, w=1e-2))
    # Capa Oculta 2
    model.add(layers.Dense(units=100, activation=activations.Relu(), regu=reg2, w=1e-2))
    # Capa de Salida
    model.add(layers.Dense(units=10, activation=activations.Linear(), regu=reg3, w=1e-1))

    model.fit(  x=x_train, y=y_train, x_test=x_test, y_test=y_test,
                bs= 50, epochs=300, 
                opt=optimizers.SGD(lr=1e-3), loss=losses.MSE(), metric=metrics.accuracy,
                plot=False, ej="8_Relu")

def ejercicio8_bis():
    print("Ejercicio 8")

    np.random.seed(14)  # Con MSE, landa = 1e-2, lr=1e-3, con std y media, 1e-2 las 2 1ras w

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Hago el flatten de los datos
    x_train = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)
    x_test = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
    y_train = y_train.reshape(y_train.size)
    y_test = y_test.reshape(y_test.size)

    # Resto la media y divido por sigma
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train  = x_train - media
    x_train /= sigma    # x_train /= 255
    x_test   = x_test  - media
    x_test  /= sigma

    # n, dim = x_train.shape  # Numero de ejemplos de training y dimension del problema


    reg1 = regularizers.L2(1e-3)
    reg2 = regularizers.L2(1e-3)
    reg3 = regularizers.L2(1e-3)


    inputt = layers.Input(x_train.shape[1])

    model = models.Network(inputt)

    # Capa Oculta 1
    model.add(layers.Dense(units=100, activation=activations.Sigmoid(), regu=reg1, w=1e-1))
    # Capa Oculta 2
    model.add(layers.Dense(units=100, activation=activations.Sigmoid(), regu=reg2, w=1e-1))
    # Capa de Salida
    model.add(layers.Dense(units=10, activation=activations.Linear(), regu=reg3, w=1e-1))

    model.fit(  x=x_train, y=y_train, x_test=x_test, y_test=y_test,
                bs= 50, epochs=300, 
                opt=optimizers.SGD(lr=1e-2), loss=losses.MSE(), metric=metrics.accuracy,
                plot=False, ej="8_Sigmoid")

#-----------------------------------------------------
# Funciones para hacer todos los graficos comparativos
#-----------------------------------------------------
def comparacion_ej3():

    data_ej3 = np.load("Informe/345/datos_ej_3.npz")
    data_Soft_Lineal = np.load("Informe/345/datos_Softmax.npz")
    data_SVM_Lineal = np.load("Informe/345/datos_SVM.npz")

    plt.plot(np.arange(len(data_ej3['Acc_t'])), data_ej3['Acc_t'], label='RN - Ej3')
    plt.plot(np.arange(len(data_Soft_Lineal['Acc_t'])) ,data_Soft_Lineal['Acc_t'], label='Lineal - Softmax')
    plt.plot(np.arange(len(data_SVM_Lineal['Acc_t'])) ,data_SVM_Lineal['Acc_t'], label='Lineal - SVM')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    # plt.savefig('Informe/Comp_EJ3.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
def comparacion_ej4():
    data_ej3 = np.load("Informe/345/datos_ej_3.npz")
    data_ej4 = np.load("Informe/345/datos_ej_4.npz")
    data_Soft_Lineal = np.load("Informe/345/datos_Softmax.npz")
    data_SVM_Lineal = np.load("Informe/345/datos_SVM.npz")

    plt.plot(np.arange(len(data_ej3['Acc_t'])), data_ej3['Acc_t'], label='RN - Ej3')
    plt.plot(np.arange(len(data_ej4['Acc_t'])), data_ej4['Acc_t'], label='RN - Ej4')
    plt.plot(np.arange(len(data_Soft_Lineal['Acc_t'])) ,data_Soft_Lineal['Acc_t'], label='Lineal - Softmax')
    plt.plot(np.arange(len(data_SVM_Lineal['Acc_t'])) ,data_SVM_Lineal['Acc_t'], label='Lineal - SVM')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ4.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def comparacion_ej5():
    data_5_RLM = np.load("Informe/345/datos_ej_5_Relu_Lin_MSe.npz")
    data_5_RLS = np.load("Informe/345/datos_ej_5_Relu_Lin_Soft.npz")
    data_5_RSM = np.load("Informe/345/datos_ej_5_Relu_Sig_MSe.npz")
    data_5_RSS = np.load("Informe/345/datos_ej_5_Relu_Sig_Soft.npz")

    plt.plot(np.arange(len(data_5_RLM['Acc_t'])), data_5_RLM['Acc_t'], label='MSE')
    plt.plot(np.arange(len(data_5_RLS['Acc_t'])), data_5_RLS['Acc_t'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc_t_Linear.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RLM['Acc'])), data_5_RLM['Acc'], label='MSE')
    plt.plot(np.arange(len(data_5_RLS['Acc'])), data_5_RLS['Acc'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Training",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc_Linear.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RLM['Loss'])), data_5_RLM['Loss'], label='MSE')
    plt.plot(np.arange(len(data_5_RLS['Loss'])), data_5_RLS['Loss'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Loss_Linear.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    #############
    plt.plot(np.arange(len(data_5_RSM['Acc_t'])), data_5_RSM['Acc_t'], label='MSE')
    plt.plot(np.arange(len(data_5_RSS['Acc_t'])), data_5_RSS['Acc_t'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc_t_Sig.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RSM['Acc'])), data_5_RSM['Acc'], label='MSE')
    plt.plot(np.arange(len(data_5_RSS['Acc'])), data_5_RSS['Acc'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Training",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc_Sig.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RSM['Loss'])), data_5_RSM['Loss'], label='MSE')
    plt.plot(np.arange(len(data_5_RSS['Loss'])), data_5_RSS['Loss'], label='Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Loss_Sig.pdf', format='pdf', bbox_inches='tight')
    plt.show()


    ################
    plt.plot(np.arange(len(data_5_RSM['Acc_t'])), data_5_RSM['Acc_t'], label='Sigmoid-MSE')
    plt.plot(np.arange(len(data_5_RSS['Acc_t'])), data_5_RSS['Acc_t'], label='Sigmoid-Softmax')
    plt.plot(np.arange(len(data_5_RLM['Acc_t'])), data_5_RLM['Acc_t'], label='Linear-MSE')
    plt.plot(np.arange(len(data_5_RLS['Acc_t'])), data_5_RLS['Acc_t'], label='Linear-Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc_t.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RSM['Acc'])), data_5_RSM['Acc'], label='Sigmoid-MSE')
    plt.plot(np.arange(len(data_5_RSS['Acc'])), data_5_RSS['Acc'], label='Sigmoid-Softmax')
    plt.plot(np.arange(len(data_5_RLM['Acc'])), data_5_RLM['Acc'], label='Linear-MSE')
    plt.plot(np.arange(len(data_5_RLS['Acc'])), data_5_RLS['Acc'], label='Linear-Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Training",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_5_RSM['Loss'])), data_5_RSM['Loss'], label='Sigmoid-MSE')
    plt.plot(np.arange(len(data_5_RSS['Loss'])), data_5_RSS['Loss'], label='Sigmoid-Softmax')
    plt.plot(np.arange(len(data_5_RLM['Loss'])), data_5_RLM['Loss'], label='Linear-MSE')
    plt.plot(np.arange(len(data_5_RLS['Loss'])), data_5_RLS['Loss'], label='Linear-Softmax')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ5_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.show()




    


def comparacion_ej6():
    data_ej6_1 = np.load("Informe/678/datos_ej_6_1.npz")
    data_ej6_2 = np.load("Informe/678/datos_ej_6_2.npz")

    plt.plot(np.arange(len(data_ej6_1['Loss'])), data_ej6_1['Loss'], label='1er Arq.')
    plt.plot(np.arange(len(data_ej6_2['Loss'])), data_ej6_2['Loss'], label='2da Arq.')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ6_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_ej6_1['Acc'])), data_ej6_1['Acc'], label='1er Arq.')
    plt.plot(np.arange(len(data_ej6_2['Acc'])), data_ej6_2['Acc'], label='2da Arq.')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ6_Acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def comparacion_ej7():

    data_ej7_1 = np.load("Informe/678/datos_ej_7_N1=10_N2=2.npz", allow_pickle=True)
    data_ej7_2 = np.load("Informe/678/datos_ej_7_N1=10_N2=10.npz", allow_pickle=True)
    data_ej7_3 = np.load("Informe/678/datos_ej_7_N1=10_N2=80.npz", allow_pickle=True)

    plt.plot(np.arange(len(data_ej7_1['Loss'])), data_ej7_1['Loss'], label='N=10, $N´=2$')
    plt.plot(np.arange(len(data_ej7_2['Loss'])), data_ej7_2['Loss'], label='N=10, $N´=10$')
    plt.plot(np.arange(len(data_ej7_3['Loss'])), data_ej7_3['Loss'], label='N=10, $N´=80$')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ7_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_ej7_1['Acc'])), data_ej7_1['Acc'], label='N=10, $N´=2$')
    plt.plot(np.arange(len(data_ej7_2['Acc'])), data_ej7_2['Acc'], label='N=10, $N´=10$')
    plt.plot(np.arange(len(data_ej7_3['Acc'])), data_ej7_3['Acc'], label='N=10, $N´=80$')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ7_Acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def comparacion_ej8():
    data_ej3 = np.load("Informe/345/datos_ej_3.npz")
    data_ej8_R = np.load("Informe/678/datos_ej_8_Relu.npz")
    data_ej8_S = np.load("Informe/678/datos_ej_8_Sigmoid.npz")
    data_Soft_Lineal = np.load("Informe/345/datos_Softmax.npz")
    data_SVM_Lineal = np.load("Informe/345/datos_SVM.npz")

    plt.plot(np.arange(len(data_ej3['Acc_t'])), data_ej3['Acc_t'], label='Ej3')
    plt.plot(np.arange(len(data_ej8_R['Acc_t'])), data_ej8_R['Acc_t'], label='Ej8 - Relu')
    plt.plot(np.arange(len(data_ej8_S['Acc_t'])), data_ej8_S['Acc_t'], label='Ej8 - Sigmoid')
    plt.plot(np.arange(len(data_Soft_Lineal['Acc_t'])) ,data_Soft_Lineal['Acc_t'], label='Lineal - Softmax')
    plt.plot(np.arange(len(data_SVM_Lineal['Acc_t'])) ,data_SVM_Lineal['Acc_t'], label='Lineal - SVM')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy Test",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ8_Acc_t.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.plot(np.arange(len(data_ej3['Acc'])), data_ej3['Acc'], label='Ej3')
    plt.plot(np.arange(len(data_ej8_R['Acc'])), data_ej8_R['Acc'], label='Ej8 - Relu')
    plt.plot(np.arange(len(data_ej8_S['Acc'])), data_ej8_S['Acc'], label='Ej8 - Sigmoid')
    plt.plot(np.arange(len(data_Soft_Lineal['Acc'])) ,data_Soft_Lineal['Acc'], label='Lineal - Softmax')
    plt.plot(np.arange(len(data_SVM_Lineal['Acc'])) ,data_SVM_Lineal['Acc'], label='Lineal - SVM')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ8_Acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.semilogy(np.arange(len(data_ej3['Loss'])), data_ej3['Loss'], label='Ej3')
    plt.semilogy(np.arange(len(data_ej8_R['Loss'])), data_ej8_R['Loss'], label='Ej8 - Relu')
    plt.semilogy(np.arange(len(data_ej8_S['Loss'])), data_ej8_S['Loss'], label='Ej8 - Sigmoid')
    plt.semilogy(np.arange(len(data_Soft_Lineal['Loss'])) ,data_Soft_Lineal['Loss'], label='Lineal - Softmax')
    plt.semilogy(np.arange(len(data_SVM_Lineal['Loss'])) ,data_SVM_Lineal['Loss'], label='Lineal - SVM')
    plt.xlabel("Epoca",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.savefig('Informe/Comp_EJ8_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    # ejercicio3()
    # comparacion_ej3()

    # ejercicio4()
    # comparacion_ej4()

    # ejercicio5()
    # comparacion_ej5()

    # ejercicio6_1()

    # ejercicio6_2()
    # comparacion_ej6()

    # ejercicio7(10,80)
    # ejercicio7(10,10)
    # ejercicio7(10,2)

    # comparacion_ej7()

    # ejercicio8()

    # ejercicio8_bis()

    comparacion_ej8()
