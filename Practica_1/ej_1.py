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

# Soy una semilla para que me cree los mismos puntos aleatorios
np.random.seed(20)

class createData():
    def __init__(self,N):
        self.N = N
        self.x = None
        self.y = None

N = 5   # Dimension
coef = np.random.randint(-5,5,(N+1,1))

M = 50   # Numero de datos
x = np.random.uniform(-100, 100, size=(M,N))

# Le tengo que meter los 1
x_ones = np.ones((M,1))

x = np.hstack((x_ones,x))   # Chequear esta


# y = coef @ x
y = x @ coef

noise = np.random.normal(0, 100, size=y.shape)

y_noise = y + noise

beta = np.linalg.inv(x.T @ x) @ x.T @ y_noise

"""
for i in range(N):
    print(i)
    plt.plot(x[:,i+1], y_noise, 'o')
    plt.plot(x[:,i+1], x[:,i+1] * beta[i+1] + beta[0], 'k')
#plt.plot(x[:,1], y, 'ob')
plt.show()
"""

plt.plot(x[:,1], y_noise, 'ob')
plt.plot(x[:,1], y, 'or')
plt.plot(x[:,1], x[:,1] * beta[1] + beta[0], '--k', label="ajuste")
plt.plot(x[:,1], x[:,1] * coef[1] + coef[0], '--g', label="posta")
plt.legend(loc="best")
plt.show()






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