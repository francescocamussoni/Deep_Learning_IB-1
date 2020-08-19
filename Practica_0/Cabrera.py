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

def ejercicio_1():
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

def ejercicio_2():
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


def ejercicio_3(a,b,c):
	res = Bhaskara(a,c,b)

	print("Ejercicio 3: Las raices de {}x²+{}x+{} son:".format(a,b,c))
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

def ejercicio_4(a,b,c):
	grafYRaices(a,b,c)

#---------------------------------
#			Ejercicio 5
#---------------------------------

class Lineal(object):
	"""docstring for Lineal"""
	def __init__(self, a,b):
		#super(Lineal, self).__init__()
		self.a = a
		self.b = b
	def __call__(self,x):
		return self.a*x+self.b


def ejercicio_5(a,b,x):
	t = Lineal(a,b)
	print("Ejercicio 5: {}x+{} en x={} es: {}".format(a,b,x,t(x)))


#---------------------------------
#			Ejercicio 6
#---------------------------------

class Exponencial(Lineal):
	"""docstring for Exponencial"""
	def __init__(self, a,b):
		super(Exponencial, self).__init__(a,b)
	def __call__(self,x):
		return self.a * x**self.b


def ejercicio_6(a,b,x):
	e = Exponencial(a,b)
	print("Ejercicio 6: {}*x^{} en x={} es: {}".format(a,b,x,e(x)))

#---------------------------------
#			Ejercicio 7
#---------------------------------

import circunferencia as circle
from circunferencia import PI, area


def ejercicio_7():
	print("Ejercicio 7:")
	print("Valor de PI importando con el alias \"circle\": {}".format(circle.PI))
	print("Valor de del area de un circulo de radio 2, importando con el alias \"circle\": {}".format(circle.area(2)))

	print("Valor de PI importando directamente: {}".format(PI))
	print("Valor de del area de un circulo de radio 2, importando directamente: {}".format(area(2)))

	if PI is circle.PI and area is circle.area:
		print("Son el mismo objeto")
	else:
		print("Son objetos distintos")

#---------------------------------
#			Ejercicio 8
#---------------------------------

# Nada que hacer aca

#---------------------------------
#			Ejercicio 9
#---------------------------------

import p0_lib
from p0_lib import rectangulo
from p0_lib.circunferencia import PI, area
from p0_lib.elipse import area
from p0_lib.rectangulo import area as area_rect

#---------------------------------
#			Ejercicio 10
#---------------------------------

def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

def ejercicio_10():
	n = 10
	x = np.linspace(-3, 3, 4 * n)
	y = np.linspace(-3, 3, 3 * n)
	X, Y = np.meshgrid(x,y)

	plt.axes([0.025, 0.025, 0.95, 0.95])
	plt.imshow(f(X, Y), cmap='bone', origin='lower')	# De los tipos de interpolacion, el unico que deja pixeleada la imagen
									# es "nearest", pero no cambia nada con dejarlo en blanco (creo).
	plt.colorbar(shrink=.8)

	plt.title("Ejercicio 10")
	plt.xticks(())
	plt.yticks(())
	#plt.savefig('Informe/ej_10.pdf', format='pdf', bbox_inches='tight')
	plt.show()

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













if __name__ == '__main__':

	ejercicio_1()

	ejercicio_2()

	ejercicio_3(1,1,1)

	ejercicio_4(-3,2,16)

	ejercicio_5(5,3,2)

	ejercicio_6(5,3,2)

	ejercicio_7()