#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19-08-2020
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

#---------------------------------
#			Ejercicio 1
#---------------------------------

A = np.array([[ 1,  0,  1],
			  [ 2, -1,  1],
			  [-3,  2, -2]])

b = np.array([[-2],
			  [ 1],
			  [-1]])

invA = np.linalg.inv(A)

res = np.dot(invA,b)

print("Ejercicio 1: res = {}".format(res))


#---------------------------------
#			Ejercicio 2
#---------------------------------

from scipy.stats import gamma

shape = 2	# Se suele notar con k
scale = 3	# Se suele notar con $\theta$


data = np.random.gamma(shape,scale,1000)

# Calculo la media y el desvio standard
mean = data.mean()		# mu = k * $\theta$
std = data.std()		# sigma² = k * $\theta^2$


x = np.linspace(0, data.max(), 1000)

# Para chequear la distribucion, obtengo k y theta a partir de los datos estadisticos y grafico
# la distribucion gamma con estos parametros.
k = (mean / std)**2
theta = (std**2) / mean

dist = gamma(a=k, scale = theta)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(data, bins=30, alpha=0.8, label="Datos")
ax.plot(x, dist.pdf(x) *1000, 'r', label="Dist. Gamma")	# Lo desnormalice
ax.legend(loc="best")
ax.set_ylabel("Cuentas")
#fig.savefig('Informe/ej_2.pdf', format='pdf', bbox_inches='tight')
plt.show()


#---------------------------------
#			Ejercicio 3
#---------------------------------

def Bhaskara(a,b=0,c=0):
	return np.roots([a,b,c]) 	# No se si esto es trampa

res = Bhaskara(1,1,1)

print("Ejercicio 3: Las raices de x²+x+1 son:")
print("x_1 = {}\nx_2 = {}".format(res[0], res[1]))