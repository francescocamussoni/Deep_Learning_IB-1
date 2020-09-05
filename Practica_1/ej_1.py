#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 29-08-2020
File: ej_1.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Doy una semilla para que me cree los mismos puntos aleatorios
np.random.seed(10)


class ajusteLineal():
    def __init__(self,N, argMin=-10, argMax=10):
        self.N = N
        self.min = argMin
        self.max = argMax
        # Para no complicarme, los coeficientes del hiperplano los hago entero en un rango razonable
        #self.coef = np.random.randint(-5,5,(N+1,1))         
        self.coef = np.random.uniform(-1,1,(N+1,1))         

    def RandomData(self, M, std=1):
        self.X = np.random.uniform(self.min, self.max, size=(M,self.N))
        self.X = np.hstack((np.ones((M,1)), self.X))   # Agrego 1 para el bias
        self.Y = self.X @ self.coef         # Estos son los y reales
        noise = np.random.normal(0, std, size=self.Y.shape)
        self.Y_noise = self.Y + noise
        # Aca saco los coeficientes ajustados
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y_noise

    def plotAll(self, save=True):
        for i in range(self.N):
            #plt.plot(self.X[:,i+1], self.Y_noise, 'o', label="Dim. "+str(i+1))
            #plt.plot(self.X[:,i+1], self.X[:,[0,i+1]] @ self.coef[[0,i+1]], 'o', label="Dim. "+str(i+1))
            plt.plot(self.X[:,i+1], self.X.T[i+1] * self.coef[i+1], 'o', label="Dim. "+str(i+1))
            #plt.plot(self.X[:,i+1], self.X[:,i+1] * self.coef[i+1] + self.coef[0], 'o', label="Dim. "+str(i+1))
            plt.plot(self.X[:,i+1], self.X[:,i+1] * self.beta[i+1] + self.beta[0], 'k')
        plt.legend(loc="best")
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        if(save):
            plt.savefig('Informe/1_Todos.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    
    def plotDim(self,i, save=True):
        plt.plot(self.X[:,i], self.Y_noise, 'ob')
        #plt.plot(self.X[:,i], self.Y, 'or')
        plt.plot(self.X[:,i], self.X[:,[0,i]] @ self.coef[[0,i]], 'o', label="Dim. "+str(i+1))
        plt.plot(self.X[:,i], self.X[:,i] * self.beta[i] + self.beta[0], '--k', label="ajuste" + str(8))
        plt.plot(self.X[:,i], self.X[:,i] * self.coef[i] + self.coef[0], '--g', label="posta")
        plt.legend(loc="best")
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        if(save):
            plt.savefig('Informe/1_UnaDimension.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    
    def getError(self):
        diff = self.coef - self.beta
        return np.linalg.norm(diff)


def barrido_datos():
    


"""
n = 100 # Numeros de datos
p = 1   # Dimension

A = np.random.uniform(-1, 1, size=(p+1,1) )

X = np.random.uniform(0, 100, size=(n,p) )

#print(X)
X = np.hstack( (np.ones((n,1)), X) )

Y = X @ A

ruido = np.random.normal(0, 1, size=Y.shape)

Y_ruido = Y + ruido


beta = np.linalg.inv(X.T @ X) @ X.T @ Y_ruido



X_plot = np.linspace(-100,100,10000)

plt.plot(X[:,1] , Y_ruido, 'ro')
plt.plot(X[:,1] , Y, 'bo')
plt.plot(X_plot , beta[0] + beta[1]*X_plot, '-y')
plt.show()
"""



"""
N = 5   # Dimension
#coef = np.random.randint(-5,5,(N+1,1))
coef = np.random.uniform(-5,5,(N+1,1))

M = 5000   # Numero de datos
x = np.random.uniform(0, 100, size=(M,N))



x = np.hstack((np.ones((M,1)),x))   # Agrego 1 para el bias


# y = coef @ x
y = x @ coef

noise = np.random.normal(0, 10, size=y.shape)

y_noise = y + noise

beta = np.linalg.inv(x.T @ x) @ x.T @ y_noise
"""

"""

for i in range(N):
    print(i)
    plt.plot(x[:,i+1], y_noise, 'o')
    plt.plot(x[:,i+1], x[:,i+1] * beta[i+1] + beta[0], 'k')
#plt.plot(x[:,1], y, 'ob')
plt.show()


plt.plot(x[:,1], y_noise, 'ob')
plt.plot(x[:,1], y, 'or')
plt.plot(x[:,1], x[:,1] * beta[1] + beta[0], '--k', label="ajuste" + str(8))
plt.plot(x[:,1], x[:,1] * coef[1] + coef[0], '--g', label="posta")
plt.legend(loc="best")
plt.show()

"""


"""

kk = ajusteLineal(5)
kk.RandomData(100)
kk.plotAll()

for i in range(2):
    kk.plotDim(i+1)

"""



# Formula para minimzar el error cuadratico medio (MCO):
# $\beta = (X^{T}X)^{-1}X^{T}Y$

#X = np.array([np.ones(M), x])
# X = np.array([np.ones(len(X)), X]).T

#y = coef @ X

#plt.plot(, )



#################
# X = np.random.uniform(size=(10,3))
# n,m = X.shape # for generality
#X0 = np.ones((1,M))
#Xnew = np.vstack((X0,x))