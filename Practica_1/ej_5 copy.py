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


class LinearClassifier():
    def __init__(self, n):
        self.n = n  #Numero de clases
        pass
    def fit(self, x_train, y_train, x_test, y_test, size_bacht=50, lr=1e-3, landa=1e-2, epochs=500, normalize=False):
        self.lr = lr
        self.l = landa
        self.epochs = epochs
        self.sbacht = size_bacht
        self.loss = np.array([])
        self.acc = np.array([])
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
            
            if (e % 100 == 0):
                print(e," ",c_acc/n_bacht)
            
            self.loss = np.append(self.loss, c_loss/n_bacht)        # Me dijeron que tome el promedio, pero no me gusta
            self.acc  = np.append(self.acc,   c_acc/n_bacht)        # Me dijeron que tome el promedio, pero no me gusta
        
        test_predict = self.predict(self.X_t)

        self.acc_test = self.accuracy(test_predict, self.Y_t)

        print("Precision final con los datos de entrenamiento: ", self.acc[-1])
        print("Precision con los datos de test: ", self.acc_test)
        
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
        return self.acc_test


class SVM(LinearClassifier):
    def __init__(self, n, delta=1):
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
        #super(SVM, self)
        super()

        scores = np.dot(self.W, X.T)                #Calculo los Scores

        scores = scores - scores.max(axis=0)        # Le resto el score maximo para que sea mas estable
        #scores -= np.max(scores, axis=0)[np.newaxis,:]

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)
        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        exp = np.exp(scores)            # Tomo las exponenciales de cada Score 

        sumatoria = exp.sum(axis=0)         # Necesitamos la sumatoria de todas las exponenciales

        softmax_fun = exp * 

        #log_softmax = np.log(sumatoria) - y_win
        log_softmax = np.log(sumatoria) - y_win             # Calculamos el log Softmax de cada imagen

        loss = log_softmax.mean() + 0.5 * self.l * np.sum(self.W * self.W)      # Promedio todas las imagenes y sumo
                                                                                # regularizacion

        inv_sumatoria = 1.0/sumatoria

        grad = inv_sumatoria *exp
        
        grad[Y, idx] -= 1

        dw = (np.dot(grad, X) / self.sbacht ) + self.l * self.W

        return loss, dw



#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
svm = SVM(10)
#svm.fit(X,Y,X_test,Y_test)
svm.fit(x_train, y_train, x_test, y_test)
"""


"""
sof = Softmax(10)
#svm.fit(X,Y,X_test,Y_test)
sof.fit(x_train, y_train, x_test, y_test)

"""

#def ejercicio_5_SVM():

np.random.seed(10)



(x_train, y_train), (x_test, y_test) = mnist.load_data()


lear_rate = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g = []
acc_g  = []
acc_t  = []

for i in lear_rate:
    svm = SVM(10)
    svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=250)

    #loss_g = np.append(loss_g, [svm.getLoss()])
    #acc_g  = np.append(acc_g, [svm.getAccuracy()])
    #acc_t  = np.append(acc_t, [svm.getAccTest()])
    loss_g += [svm.getLoss()]
    acc_g += [svm.getAccuracy()]
    acc_t += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g[0], label=r"lr=1e-1")
plt.plot(e, loss_g[1], label=r"lr=1e-2")
plt.plot(e, loss_g[2], label=r"lr=1e-3")
plt.plot(e, loss_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Loss_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g[0], label=r"lr=1e-1")
plt.plot(e, acc_g[1], label=r"lr=1e-2")
plt.plot(e, acc_g[2], label=r"lr=1e-3")
plt.plot(e, acc_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Acc_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

# Ahora pruebo variando lanmda
lambda_v = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g_2 = []
acc_g_2  = []
acc_t_2  = []

for i in lambda_v:
    svm = SVM(10)
    svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=250)

    # loss_g_2 = np.append(loss_g, [svm.getLoss()])
    # acc_g_2  = np.append(loss_g, [svm.getAccuracy()])
    # acc_t_2  = np.append(loss_g, [svm.getAccTest()])

    loss_g_2 += [svm.getLoss()]
    acc_g_2 += [svm.getAccuracy()]
    acc_t_2 += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g_2[0], label=r"$\lambda=1$")
plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, loss_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Loss_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g_2[0], label=r"$\lambda=1$")
plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, acc_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Acc_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()







#########################################################



lear_rate = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g = []
acc_g  = []
acc_t  = []

for i in lear_rate:
    svm = Softmax(10)
    svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=250)

    #loss_g = np.append(loss_g, [svm.getLoss()])
    #acc_g  = np.append(acc_g, [svm.getAccuracy()])
    #acc_t  = np.append(acc_t, [svm.getAccTest()])
    loss_g += [svm.getLoss()]
    acc_g += [svm.getAccuracy()]
    acc_t += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g[0], label=r"lr=1e-1")
plt.plot(e, loss_g[1], label=r"lr=1e-2")
plt.plot(e, loss_g[2], label=r"lr=1e-3")
plt.plot(e, loss_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Loss_Soft_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g[0], label=r"lr=1e-1")
plt.plot(e, acc_g[1], label=r"lr=1e-2")
plt.plot(e, acc_g[2], label=r"lr=1e-3")
plt.plot(e, acc_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Acc_Soft_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

# Ahora pruebo variando lanmda
lambda_v = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g_2 = []
acc_g_2  = []
acc_t_2  = []

for i in lambda_v:
    svm = Softmax(10)
    svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=250)

    # loss_g_2 = np.append(loss_g, [svm.getLoss()])
    # acc_g_2  = np.append(loss_g, [svm.getAccuracy()])
    # acc_t_2  = np.append(loss_g, [svm.getAccTest()])

    loss_g_2 += [svm.getLoss()]
    acc_g_2 += [svm.getAccuracy()]
    acc_t_2 += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g_2[0], label=r"$\lambda=1$")
plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, loss_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Loss_Soft_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g_2[0], label=r"$\lambda=1$")
plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, acc_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Acc_Soft_MNIST.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()





###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################





"""


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


lear_rate = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g = []
acc_g  = []
acc_t  = []

for i in lear_rate:
    svm = SVM(10)
    svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=250)

    #loss_g = np.append(loss_g, [svm.getLoss()])
    #acc_g  = np.append(acc_g, [svm.getAccuracy()])
    #acc_t  = np.append(acc_t, [svm.getAccTest()])
    loss_g += [svm.getLoss()]
    acc_g += [svm.getAccuracy()]
    acc_t += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g[0], label=r"lr=1e-1")
plt.plot(e, loss_g[1], label=r"lr=1e-2")
plt.plot(e, loss_g[2], label=r"lr=1e-3")
plt.plot(e, loss_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Loss.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g[0], label=r"lr=1e-1")
plt.plot(e, acc_g[1], label=r"lr=1e-2")
plt.plot(e, acc_g[2], label=r"lr=1e-3")
plt.plot(e, acc_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

# Ahora pruebo variando lanmda
lambda_v = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g_2 = []
acc_g_2  = []
acc_t_2  = []

for i in lambda_v:
    svm = SVM(10)
    svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=250)

    # loss_g_2 = np.append(loss_g, [svm.getLoss()])
    # acc_g_2  = np.append(loss_g, [svm.getAccuracy()])
    # acc_t_2  = np.append(loss_g, [svm.getAccTest()])

    loss_g_2 += [svm.getLoss()]
    acc_g_2 += [svm.getAccuracy()]
    acc_t_2 += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g_2[0], label=r"$\lambda=1$")
plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, loss_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Loss.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g_2[0], label=r"$\lambda=1$")
plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, acc_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Acc.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()







#########################################################



lear_rate = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g = []
acc_g  = []
acc_t  = []

for i in lear_rate:
    svm = Softmax(10)
    svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=250)

    #loss_g = np.append(loss_g, [svm.getLoss()])
    #acc_g  = np.append(acc_g, [svm.getAccuracy()])
    #acc_t  = np.append(acc_t, [svm.getAccTest()])
    loss_g += [svm.getLoss()]
    acc_g += [svm.getAccuracy()]
    acc_t += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g[0], label=r"lr=1e-1")
plt.plot(e, loss_g[1], label=r"lr=1e-2")
plt.plot(e, loss_g[2], label=r"lr=1e-3")
plt.plot(e, loss_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Loss_Soft.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g[0], label=r"lr=1e-1")
plt.plot(e, acc_g[1], label=r"lr=1e-2")
plt.plot(e, acc_g[2], label=r"lr=1e-3")
plt.plot(e, acc_g[3], label=r"lr=1e-4")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_Lr_Acc_Soft.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

# Ahora pruebo variando lanmda
lambda_v = [1e-1, 1e-2, 1e-3, 1e-4]
loss_g_2 = []
acc_g_2  = []
acc_t_2  = []

for i in lambda_v:
    svm = Softmax(10)
    svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=250)

    # loss_g_2 = np.append(loss_g, [svm.getLoss()])
    # acc_g_2  = np.append(loss_g, [svm.getAccuracy()])
    # acc_t_2  = np.append(loss_g, [svm.getAccTest()])

    loss_g_2 += [svm.getLoss()]
    acc_g_2 += [svm.getAccuracy()]
    acc_t_2 += [svm.getAccTest()]

loss_g = np.array(loss_g)
acc_g = np.array(acc_g)
acc_t = np.array(acc_t)

e = np.arange(250)

plt.plot(e, loss_g_2[0], label=r"$\lambda=1$")
plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, loss_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Funcion de coste", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Loss_Soft.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

plt.plot(e, acc_g_2[0], label=r"$\lambda=1$")
plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-1$")
plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-2$")
plt.plot(e, acc_g_2[3], label=r"$\lambda=1e-3$")
plt.xlabel("Epoca", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.savefig('Informe/5_lambda_Acc_Soft.pdf', format='pdf', bbox_inches='tight')
plt.close()
#plt.show()

"""

