#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 02-09-2020
File: ej_5.py
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
import ipdb


class LinearClassifier():
    def __init__(self, n):
        self.n = n  #Numero de clases
        pass
    def fit(self, x_train, y_train, x_test, y_test, size_bacht=50, lr=1e-3, landa=1e-5, epochs=200, normalize=False):
        self.lr = lr
        self.l = landa
        self.epochs = epochs
        self.sbacht = size_bacht
        self.loss = np.array([])
        self.acc = np.array([])
        self.test_acc = np.array([])
        # Tuve algunos problemas con el pasaje de argumentos por referencia, por el momento esto lo arregla
        self.X = np.copy(x_train)       
        self.X_t = np.copy(x_test)

        # Acomodo las entradas del entrenamiento
        self.X = self.X.reshape(len(self.X), self.X[0].size).astype(np.float)
        self.X = np.hstack((np.ones((len(self.X),1)), self.X))              # Pongo 1 para el bias
        self.Y = y_train.reshape(y_train.size)

        # Acomodo las entradas del test
        self.X_t = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
        self.X_t = np.hstack((np.ones((len(self.X_t),1)), self.X_t))        # Pongo 1 para el bias
        self.Y_t = y_test.reshape(y_test.size)

        if(normalize):
            self.X   /= 255
            self.X_t /= 255

        # Inicializo W
        self.W = np.random.uniform(-10, 10, size=(self.n, self.X.shape[1]))

        n_bacht = int(len(x_train)/self.sbacht)  # Cuantos bacht tengo, pueden que me queden imagenes afuera. Arreglar

        # Recorro las epocas        
        for e in range(self.epochs):
            c_loss = 0
            c_acc = 0
            for i in range(n_bacht):

                x_bacht = self.X[self.sbacht*i: self.sbacht*(i+1)]      # Me quedo con un pedazo de los datos
                y_bacht = self.Y[self.sbacht*i: self.sbacht*(i+1)]

                loss, dw = self.loss_gradient(x_bacht, y_bacht)     # Me molesta que el VSCode me tire que esta linea esta mal

                c_loss += loss

                predict_y = self.predict(x_bacht)
                c_acc += self.accuracy(y_bacht, predict_y)
                
                self.W -= self.lr * dw
            
            if (e % 10 == 0):
                print(e,"/",self.epochs," ",c_acc/n_bacht)
            
            self.loss = np.append(self.loss, c_loss/n_bacht)        # Me dijeron que tome el promedio, pero no me gusta
            self.acc  = np.append(self.acc,   c_acc/n_bacht)        # Me dijeron que tome el promedio, pero no me gusta
            predict_test = self.predict(self.X_t)
            self.test_acc = np.append(self.test_acc, self.accuracy(self.Y_t, predict_test))
        
        #test_predict = self.predict(self.X_t)

        #self.acc_test = self.accuracy(test_predict, self.Y_t)

        print("Precision final con los datos de entrenamiento: ", self.acc[-1])
        print("Precision final con los datos de test: ", self.test_acc[-1])
        
        self.e = np.arange(self.epochs)

        #self.plotLoss()
        #self.plotAcurracy()

    def plotLoss(self):

        plt.plot(self.e, self.loss)
        plt.show()

    def plotAcurracy(self):

        plt.plot(self.e, self.loss)
        plt.show()

    def predict(self, x_testt):
        if(self.X.shape[1] != x_testt.shape[1]):
            x_testt = x_testt.reshape(len(x_testt), x_testt[0].size).astype(np.float)
            x_testt = np.hstack((np.ones((len(x_testt),1)), x_testt))
        scores = np.dot(self.W, x_testt.T)
        return np.argmax(scores, axis=0)

    def loss_gradient(self,X,Y):
        pass

    def accuracy(self, y_true, y_pred):
        accuracy = (np.sum(y_true == y_pred) / len(y_true))
        return accuracy*100
    
    def getLoss(self):      # Por si quiero graficar afuera
        return self.loss
    
    def getAccuracy(self):  # Por si quiero graficar afuera
        return self.acc
    
    def getAccTest(self):
        return self.test_acc[-1]
    
    def getAccuracyTest(self):
        return self.test_acc


class SVM(LinearClassifier):
    def __init__(self, n,delta=1):
        super(SVM, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        # Las dimensiones del batch ya deberian estar acomodadas

        scores = np.dot(self.W, X.T)         #Calculo los Scores

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)

        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        # Intento restar cada score a su columna (ie imagen) correspondiente
        resta = scores - y_win[np.newaxis,:] + self.delta
        resta[Y, idx] = 0                   # y acomodo a 0 los que deberian ganar
        resta[resta<0] = 0                  # Mato todo lo que es negativo

        loss = (resta.sum() / Y.size) + 0.5 * self.l * np.sum(self.W * self.W)

        resta[resta>0] = 1                  # Lo que queda es solo positivo y lo mando a 1

        resta[Y, idx] -= resta.sum(axis=0)[idx]     # El lugar debe ganar tiene la resta de todos los que le ganaron

        dw = (np.dot(resta, X) / self.sbacht ) + self.l * self.W

        return loss, dw
    

class Softmax(LinearClassifier):
    def __init__(self, n, delta=1):
        super(Softmax, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        super()

        scores = np.dot(self.W, X.T)                #Calculo los Scores

        scores = scores - scores.max(axis=0)        # Le resto el score maximo para que sea mas estable

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)
        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        exp = np.exp(scores)            # Tomo las exponenciales de cada Score 

        sumatoria = exp.sum(axis=0)         # Necesitamos la sumatoria de todas las exponenciales

        softmax_fun = exp * (1.0/sumatoria)     # Calculo la Softmax de cada imagen

        softmax_fun[Y, idx] -= 1               # Le resto la delta de Kronecker a las que tienen que ganar

        #log_softmax = np.log(sumatoria) - y_win
        log_softmax = np.log(sumatoria) - y_win             # Calculamos el log Softmax de cada imagen
                                                            # Creo que no es estrictamente la softmax, pero se entiende

        loss = log_softmax.mean() + 0.5 * self.l * np.sum(self.W * self.W)      # Promedio todas las imagenes y sumo
                                                                                # regularizacion

        #inv_sumatoria = 1.0/sumatoria

        #grad = inv_sumatoria *exp
        
        #grad[Y, idx] -= 1

        #ipdb.set_trace(context=15)

        dw = (np.dot(softmax_fun, X) / self.sbacht ) + self.l * self.W      # Me hace ruido este producto, pero
                                                            # por las dimensiones es lo unico que tiene sentido

        return loss, dw




def ejercicio_5_Softmax(c='mnist', epocas=100):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    soft = Softmax(10)

    print("Sofmax - {}".format(c))

    soft.fit(x_train, y_train, x_test, y_test, epochs=epocas, normalize=False)

    e = np.arange(epocas)
    coste = soft.getLoss()

    plt.plot(e, coste)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/Loss_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

    acc = soft.getAccuracy()

    plt.plot(e, acc)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/Accuracy_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

    acc_t = soft.getAccuracyTest()

    plt.plot(e, acc_t)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/AccuracyTest_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

def ejercicio_5_SVM(c='mnist', epocas=100):
    np.random.seed(10)


    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    svm = SVM(10)

    print("SVM - {}".format(c))

    svm.fit(x_train, y_train, x_test, y_test, epochs=epocas, normalize=False)

    e = np.arange(epocas)
    coste = svm.getLoss()

    plt.plot(e, coste, label=r"lr=1e-1")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/SVM_Loss_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

    acc = svm.getAccuracy()

    plt.plot(e, acc)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/SVM_Accuracy_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

    acc_t = svm.getAccuracyTest()

    plt.plot(e, acc_t)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/SVM_AccuracyTest_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.show()

def ejercicio_5_Comparar(c='mnist', epocas=100):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    ###### Softmax
    soft = Softmax(10)

    print("Sofmax - {}".format(c))

    soft.fit(x_train, y_train, x_test, y_test, epochs=epocas)

    softmax_coste = soft.getLoss()
    softmax_acc   = soft.getAccuracy()
    softmax_acc_test   = soft.getAccuracyTest()

    ###### SVM
    svm = SVM(10)

    print("SVM - {}".format(c))

    svm.fit(x_train, y_train, x_test, y_test, epochs=epocas)

    svm_coste = svm.getLoss()
    svm_acc   = svm.getAccuracy()
    svm_acc_test   = svm.getAccuracyTest()
    
    #### GRAFICO

    e = np.arange(epocas)

    plt.figure()

    plt.plot(e, softmax_coste, label=r"Softmax")
    plt.plot(e, svm_coste, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/Comparacion_Coste_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    plt.close()
    #plt.show()

    plt.figure()

    plt.plot(e, softmax_acc, label=r"Softmax")
    plt.plot(e, svm_acc, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Entrenamiento", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/Comparacion_Precision_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    plt.close()
    #plt.show()

    plt.figure()

    plt.plot(e, softmax_acc_test, label=r"Softmax")
    plt.plot(e, svm_acc_test, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Informe/5/Comparacion_Precision_TEST_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    plt.close()
    #plt.show()



ejercicio_5_Softmax(c='mnist',epocas=200)
ejercicio_5_SVM(c='mnist',epocas=200)

ejercicio_5_Comparar(c='mnist',epocas=200)
ejercicio_5_Comparar(c='cifar10', epocas=200)






def barridoParametros(method='SVM',c='mnist', epocas=200):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    lear_rate = [1e-2, 1e-3, 1e-4, 1e-5]
    loss_g = []
    acc_g  = []
    acc_t  = []

    for i in lear_rate:
        if(method == 'SVM'):
            svm = SVM(10)
        elif(method == 'Softmax'):
            svm = Softmax(10)
        svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=epocas)

        loss_g += [svm.getLoss()]
        acc_g += [svm.getAccuracy()]
        acc_t += [svm.getAccuracyTest()]

    loss_g = np.array(loss_g)
    acc_g = np.array(acc_g)
    acc_t = np.array(acc_t)

    e = np.arange(epocas)

    plt.figure()
    plt.plot(e, loss_g[0], label=r"lr=1e-2")
    plt.plot(e, loss_g[1], label=r"lr=1e-3")
    plt.plot(e, loss_g[2], label=r"lr=1e-4")
    plt.plot(e, loss_g[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lr_loss_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()

    plt.figure()
    plt.plot(e, acc_g[0], label=r"lr=1e-2")
    plt.plot(e, acc_g[1], label=r"lr=1e-3")
    plt.plot(e, acc_g[2], label=r"lr=1e-4")
    plt.plot(e, acc_g[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Training", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lr_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()

    plt.figure()
    plt.plot(e, acc_t[0], label=r"lr=1e-2")
    plt.plot(e, acc_t[1], label=r"lr=1e-3")
    plt.plot(e, acc_t[2], label=r"lr=1e-4")
    plt.plot(e, acc_t[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lr_accTest_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()

    

    # Ahora pruebo variando lambda
    lambda_v = [1e-2, 1e-3, 1e-4]
    loss_g_2 = []
    acc_g_2  = []
    acc_t_2  = []

    for i in lambda_v:
        if(method == 'SVM'):
            svm = SVM(10)
        elif(method == 'Softmax'):
            svm = Softmax(10)
        svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=epocas)

        loss_g_2 += [svm.getLoss()]
        acc_g_2 += [svm.getAccuracy()]
        acc_t_2 += [svm.getAccuracyTest()]

    loss_g = np.array(loss_g)
    acc_g = np.array(acc_g)
    acc_t = np.array(acc_t)

    e = np.arange(epocas)

    plt.figure()
    plt.plot(e, loss_g_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()

    plt.figure()
    plt.plot(e, acc_g_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Training", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()

    plt.figure()
    plt.plot(e, acc_t_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, acc_t_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, acc_t_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    #plt.pause(1)
    #plt.show()


barridoParametros(method='SVM'    ,c='mnist'  ,epocas=250)
barridoParametros(method='Softmax',c='mnist'  ,epocas=250)
barridoParametros(method='SVM'    ,c='cifar10',epocas=250)
barridoParametros(method='Softmax',c='cifar10',epocas=250)


