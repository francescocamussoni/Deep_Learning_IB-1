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
ax.set_title('Ejercicio 2')
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


#---------------------------------
#			Ejercicio 4
#---------------------------------

def seteaGrilla():
	plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
	plt.grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.75')
	plt.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.75')
	plt.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')

# Defino una funcion que dados los coeficientes de un polinomio, lo grafica entre -4.5, 4 e indica sus
# raices si se encuentran en ese rango (y si son reales)
def grafYRaices(a,b,c):
	x = np.linspace(-4.5, 4, 1000)
	p = np.polyval([a,b,c], x)
	r1, r2 = Bhaskara(a,b,c)

	plt.plot(x,p,'b-')
	if r1.imag == 0 and r1 <= 4 and r1 >= -4.5:
		plt.plot(r1,0,'or')
		plt.annotate(s=r"$x_1=${:.2f}".format(r1), xy=(r1, 0), xytext=(r1+0.3, 0))
	if r2.imag == 0 and r2 <= 4 and r2 >= -4.5:
		plt.plot(r2,0,'or')
		plt.annotate(s=r"$x_2=${:.2f}".format(r2), xy=(r2, 0), xytext=(r2+0.3, 0))

	seteaGrilla()
	plt.xlabel("x")
	plt.ylabel(r"y(x) = ${}x^2 +{}x +{}$".format(a,b,c))
	plt.title("Ejercicio 4")
	#plt.savefig('Informe/ej_4.pdf', format='pdf', bbox_inches='tight')
	plt.show()

grafYRaices(-3,2,16)

#---------------------------------
#			Ejercicio 5
#---------------------------------



#---------------------------------
#			Ejercicio 6
#---------------------------------



#---------------------------------
#			Ejercicio 7
#---------------------------------



#---------------------------------
#			Ejercicio 8
#---------------------------------



#---------------------------------
#			Ejercicio 9
#---------------------------------



#---------------------------------
#			Ejercicio 10
#---------------------------------



#---------------------------------
#			Ejercicio 11
#---------------------------------



#---------------------------------
#			Ejercicio 12
#---------------------------------



#---------------------------------
#			Ejercicio 13
#---------------------------------



#---------------------------------
#			Ejercicio 14
#---------------------------------




#---------------------------------
#			Ejercicio 15
#---------------------------------