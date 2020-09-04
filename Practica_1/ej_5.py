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

"""
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})
"""

np.random.seed(10)

class LinearClassifier():
    def __init__(self, n):
        self.n = n  #Numero de clases
        pass
    def fit(self, x_train, y_train, x_test, y_test, size_bacht=50, lr=1e-4, landa=0.001, epochs=500):
        self.lr = lr
        self.l = landa
        self.epochs = epochs
        self.sbacht = size_bacht
        self.loss = np.array([])
        self.acc = np.array([])

        # Acomodo las entradas del entrenamiento
        self.X = x_train.reshape(len(x_train), x_train[0].size).astype(np.float)#/255
        self.X = np.hstack((np.ones((len(self.X),1)), self.X))              # Pongo 1 para el bias
        self.Y = y_train.reshape(y_train.size)

        # Acomodo las entradas del test
        self.X_t = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)#/255
        self.X_t = np.hstack((np.ones((len(self.X_t),1)), self.X_t))        # Pongo 1 para el bias
        self.Y_t = y_test.reshape(y_test.size)

        # Inicializo W
        self.W = np.random.uniform(-10, 10, size=(self.n, self.X.shape[1]))

        n_bacht = int(len(x_train)/self.sbacht)  # Cuantos bacht tengo

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
            
            print(e," ",c_acc/n_bacht)
            
            self.loss = np.append(self.loss, c_loss/n_bacht)
            self.acc  = np.append(self.acc,   c_acc/n_bacht)
        
        test_predict = self.predict(self.X_t)

        print("Precision final con los datos de entrenamiento: ", self.acc[-1])
        print("Precision con los datos de test: ", self.accuracy(test_predict, self.Y_t))
        

        self.e = np.arange(self.epochs)

        self.plotLoss()
        self.plotAcurracy()

    def plotLoss(self):

        plt.plot(self.e, self.loss)
        plt.show()

    def plotAcurracy(self):

        plt.plot(self.e, self.loss)
        plt.show()

    def predict(self, x_testt):
        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
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
    
    def getLoss(self):
        return self.loss
    
    def getAccuracy(self):
        return self.acc


class SVM(LinearClassifier):
    def __init__(self, n, delta=1):
        super(SVM, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        #super(SVM, self)
        super()
        # Las dimensiones del batch ya deberian estar acomodadas

        scores = np.dot(self.W, X.T)         #Calculo los Scores

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)

        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        # Intento restar cada score a su columna (ie imagen) correspondiente
        resta = scores - y_win[np.newaxis,:] + self.delta
        resta[Y, idx] = 0                   # y acomodo los 0 de los que deberian ganar
        resta[resta<0] = 0

        L = resta.sum(axis=0)
        loss = np.mean(L) + 0.5 * self.l * np.sum(self.W * self.W)


        #loss = (resta.sum() / Y.size) + 0.5 * self.l * np.sum(self.W * self.W)

        resta[resta>0] = 1

        resta[Y, idx] -= resta.sum(axis=0)[idx]

        dw = (np.dot(resta, X) / self.sbacht ) + self.l * self.W

        return loss, dw
    
    def loss_gradient2(self,x,y):
        super()
        self.delta=1#-1
        self.lambda_L2 = 0.5
        L2= np.sum(self.W*self.W)

        id= np.arange(x.shape[0], dtype=np.int)
        #yp=self.activacion(x)
        yp = np.dot(self.W, x.T)
        y=y.reshape(x.shape[0]) #por sino es como yo quiero

        diff = yp - yp[y,id] + self.delta
        diff = np.maximum(diff, 0)
        diff[y, np.arange(x.shape[0])]=0 

        #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        L=diff.sum(axis=0)
        loss = np.mean(L) + 0.5*self.lambda_L2*L2

        # 'y' tiene las posiciones de la solucion 
        # es genial porque las puedo usar para forzar el 0 donde debe ir
        diff=np.heaviside(diff,0)
        diff[y, id] -= diff.sum(axis=0)[id]

        dW = np.dot(diff, x)/x.shape[0] + self.lambda_L2*self.W
        return loss, dW


class Softmax(LinearClassifier):
    def __init__(self, n, delta=1):
        super(Softmax, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        #super(SVM, self)
        super()

        scores = np.dot(self.W, X.T)         #Calculo los Scores

        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

        # Le resto el score maximo para que sea mas estable
        #scores = scores - scores.max(axis=0)
        scores -= np.max(scores, axis=0)[np.newaxis,:]

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)

        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        exp = np.exp(scores)

        sumatoria = exp.sum(axis=0)

        L = np.log(sumatoria) - y_win

        loss = L.mean() + 0.5 * self.l * np.sum(self.W * self.W)




        ################################################
        inv_sumatoria = 1.0/sumatoria

        grad = inv_sumatoria *exp
        
        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

        grad[Y, idx] -= 1

        dw = (np.dot(grad, X) / self.sbacht ) + self.l * self.W

        return loss, dw

        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT










# Numero de clases
#n = 10
#n_samples = 4

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()


svm = SVM(10)
#svm.fit(X,Y,X_test,Y_test)
svm.fit(x_train, y_train, x_test, y_test)



"""
sof = Softmax(10)
#svm.fit(X,Y,X_test,Y_test)
sof.fit(x_train, y_train, x_test, y_test)
"""



"""
X = x_train[:n_samples]
Y = y_train[:n_samples]

X_test = x_test[:20]
Y_test = y_test[:20]

x_train = x_train.reshape(len(x_train), x_train[0].size)
y_train = y_train.reshape( y_train.size)

svm = SVM(10)
#svm.fit(X,Y,X_test,Y_test)
svm.fit(x_train, y_train, x_test, y_test)


Y_test = Y_test.reshape(Y_test.size)

X_test = X_test.reshape(len(X_test), X[0].size)
X_test = np.hstack((np.ones((len(X_test),1)), X_test))

a = svm.predict(X_test)
"""








"""
delta = 1
landa = 1

#X.shape = (#ejemplos, dim(x)+1)
#W.shape = (#clases, dim(x)+1)
#Scores = np.dot(X,W.T)


X = X.reshape(len(X), X[0].size)        # Reacomodo la entrada                                  LISTO

X = np.hstack((np.ones((len(X),1)), X))  # Pongo 1 para el bias                                 LISTO

W = np.random.uniform(0, 1, size=(n, X.shape[1]))   # Invento la W                              LISTO

Scores = np.dot(W, X.T)         #Calculo los Scores                                             LISTO



# Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)

# El primer indice es el Score que debe ganar y el segundo el numero de imagen
idx = np.arange(0,n_samples)                                                                     #LISTO

Y = Y.reshape(Y.size)           # Aca tuve que reacomodar Y porque tiene un formato feo          LISTO

# Creo que con esto deberia poder quedarme con los scores que deben ganar
y_win = Scores[Y, idx]                                                                          #LISTO


# Intento restar cada score a su columna (ie imagen) correspondiente

resta = Scores - y_win[np.newaxis,:] + delta                                                    #LISTO

#import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

#resta = np.maximum(resta, 0)        # Ahora mato todo lo que no sea positivo                    #LISTO
resta2 = np.heaviside(resta,0)

#resta[Y, idx] = 0           # y acomodo los 0 de los que deberian ganar                         #LISTO
resta2[Y, idx] = 0           # y acomodo los 0 de los que deberian ganar                         #LISTO
print(resta)                                                                                    #LISTO 



#resta[resta>0] = 1                                                                              #LISTO
# import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

# ME FALTA SUMAR TODA UNA COLUMNA Y PONERLA EN EL LUGAR QUE TIENE QUE GANAR!!!!!!!!!!!!!!!!!!!!

resta[Y, idx] -= resta.sum(axis=0)[idx]
#resta[Y,:] -= resta.sum(axis=0)



dw = (np.dot(resta, X) / n ) + landa * W
"""






"""
Metrica accuracy
"""