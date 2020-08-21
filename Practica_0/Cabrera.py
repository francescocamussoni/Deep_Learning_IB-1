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

import numpy as np
from matplotlib import pyplot as plt
import random


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

#import circunferencia as circle
#from circunferencia import PI, area

def ejercicio_7():

	import circunferencia as circle
	from circunferencia import PI, area

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


def ejercicio_11():
	n = 256
	x = np.linspace(-3, 3, n)
	y = np.linspace(-3, 3, n)
	X, Y = np.meshgrid(x,y)

	plt.figure()
	plt.axes([0.025, 0.025, 0.95, 0.95])
	plt.contourf(X, Y, f(X, Y), 7, alpha=.75, cmap=plt.cm.hot)
	C = plt.contour(X, Y, f(X, Y), 7, colors='black')
	plt.clabel(C, inline=5, fontsize=8)

	#plt.title("Ejercicio 11")
	plt.xticks(())
	plt.yticks(())
	#plt.savefig('Informe/ej_11.png', format='png', bbox_inches='tight')
	plt.show()

#---------------------------------
#			Ejercicio 12
#---------------------------------

def ejercicio_12():
	n = 1024
	X = np.random.normal(0,1,n)
	Y = np.random.normal(0,1,n)
	T = np.arctan2(Y, X)
	#T = 2 * np.pi * np.random.rand(n)

	plt.axes([0.025, 0.025, 0.95, 0.95])
	plt.scatter(X, Y, c=T, cmap='jet', alpha=0.5, edgecolors='gray')  	# https://matplotlib.org/tutorials/colors/colormaps.html

	plt.xlim(-1.5, 1.5)
	plt.xticks(())
	plt.ylim(-1.5, 1.5)
	plt.yticks(())
	#plt.savefig('Informe/ej_12.png', format='png', bbox_inches='tight')

	plt.show()


#--------------------------------------------------------------
#			Ejercicio 12 con las proyecciones en los ejes
#--------------------------------------------------------------

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y, X)

# Esta funcion esta sacada de https://matplotlib.org/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
# Pero esta un poco modificada para el ejercicio
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    #ax.tick_params(axis="x", labelbottom=False)
    #ax.tick_params(axis="y", labelleft=False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax_histx.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)


    #ax_histx.tick_params(axis="x", labelbottom=False)
    #ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(X, Y, c=T, cmap='jet', alpha=0.5, edgecolors='gray')  	# https://matplotlib.org/tutorials/colors/colormaps.html

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(X)), np.max(np.abs(Y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(X, bins=bins)
    ax_histy.hist(Y, bins=bins, orientation='horizontal')


def ejercicio_12_histograma():
	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	spacing = 0.005

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom + height + spacing, width, 0.2]
	rect_histy = [left + width + spacing, bottom, 0.2, height]

	fig = plt.figure(figsize=(8, 8))

	ax = fig.add_axes(rect_scatter)
	ax_histx = fig.add_axes(rect_histx, sharex=ax)
	ax_histy = fig.add_axes(rect_histy, sharey=ax)

	scatter_hist(X, Y, ax, ax_histx, ax_histy)

	#plt.savefig('Informe/ej_12_Histograma.pdf', format='pdf', bbox_inches='tight')

	plt.show()



#---------------------------------
#			Ejercicio 13
#---------------------------------

DT = 1
NUM_PECES = 16
PLOTEA = 1
NUM_ITERAC = 200


class R2(object):
	"""Representa un punto en R2"""
	def __init__(self, x,y):
		self.r = np.array([x,y])
	def __add__(self, R):
		aux = R2(self.r[0], self.r[1])
		aux.r = self.r + R.r
		return aux
	def __sub__(self, R):
		aux = R2(self.r[0], self.r[1])
		aux.r = self.r - R.r
		return aux
	def __truediv__(self,n):
		aux = R2(self.r[0], self.r[1])
		aux.r = self.r / n
		return aux
	def __mul__(self,n):
		aux = R2(self.r[0], self.r[1])
		aux.r = self.r * n
		return aux
	def norma(self):
		return np.linalg.norm(self.r)

class Pez(object):
	"""Representa un Pez con posicion y velocidad"""
	def __init__(self, pos, vel):
		self.pos = pos
		self.vel = vel
	def __add__(self,pez):
		aux = Pez(self.pos, self.vel)
		aux.pos = self.pos + pez.pos
		aux.vel = self.vel + pez.vel
		return aux

class Cardumen(object):
	"""docstring for Cardumen"""
	def __init__(self):
		self.N = NUM_PECES			# Cantidad de peces en el cardumen
		self.dt = DT 				# Variacion temporal
		self.iteracion = 0 			# numero de iteraciones
		self.dim = None				# Dimension de la pecera/cuadro donde estan los peces
		self.__maxVel = None
		self.__maxDist = None
		self.Peces = np.array( [Pez(R2(0,0), R2(0,0)) for i in range(self.N)] ) 
		self.CM = None 				# Centro del cardumen
		self.VM = None				# Velocidad media del cardumen
		self.gif = []

		self.fig = plt.figure()

	def CMyVM(self):
		aux = self.Peces.sum()
		self.CM = (aux.pos / self.N)
		self.VM = (aux.vel / self.N)

	def initialize(self, dim, maxVel, maxDist):
		self.dim = dim
		self.__maxVel = maxVel
		self.__maxDist = maxDist

		for i in range(self.N):
			v = random.uniform(0,maxVel)
			t = 2*np.pi*random.random()
			pez = Pez( R2(random.uniform(0,self.dim), random.uniform(0,self.dim)) , R2(v*np.cos(t), v*np.sin(t)))
			self.Peces[i] = pez

		self.CMyVM()

		plt.title("Ejercicio 13 - Cardumen - {} peces".format(self.N))
		plt.xlim(0,dim)
		plt.ylim(0,dim)
		plt.xticks(())
		plt.yticks(())		

	def reglaA(self,i):
		return (self.CM - self.Peces[i].pos) / 8

	def reglaB(self,i):
		aux = R2(0,0)
		for j in range(self.N):
			if j != i:
				dif = self.Peces[i].pos - self.Peces[j].pos
				d = dif.norma()
				if d < self.__maxDist:
					aux = aux + (dif / d)
		return aux

	def reglaC(self,i):
		return (self.VM - self.Peces[i].vel) / 8

	def reglas(self,i):
		return self.reglaA(i) + self.reglaB(i) + self.reglaC(i)

	def doStep(self):
		# Calculo los delta de velocidad de cada pez
		DeltaV = np.array([self.reglas(i) for i in range(self.N)])
		
		for i in range(self.N):
			# Modifico las posiciones y velocidades
			self.Peces[i].pos = self.Peces[i].pos + self.Peces[i].vel * self.dt
			self.Peces[i].vel = self.Peces[i].vel + DeltaV[i]

			if 		self.Peces[i].pos.r[0] > self.dim:						# Pared derecha
					self.Peces[i].pos.r[0] = 2*self.dim - self.Peces[i].pos.r[0]
					self.Peces[i].vel.r[0] = -self.Peces[i].vel.r[0]	
			elif self.Peces[i].pos.r[0] < 0:								# Pared izquierda
					self.Peces[i].pos.r[0] = - self.Peces[i].pos.r[0]
					self.Peces[i].vel.r[0] = -self.Peces[i].vel.r[0]

			if self.Peces[i].pos.r[1] > self.dim:							# Pared de arriba
					self.Peces[i].pos.r[1] = 2*self.dim - self.Peces[i].pos.r[1]
					self.Peces[i].vel.r[1] = -self.Peces[i].vel.r[1]
			elif self.Peces[i].pos.r[1] < 0:								# Pared de abajo
					self.Peces[i].pos.r[1] = - self.Peces[i].pos.r[1]
					self.Peces[i].vel.r[1] = -self.Peces[i].vel.r[1]		# Lo hago rebotar

			# Limito la velocidad
			if self.Peces[i].vel.norma() > self.__maxVel:
					self.Peces[i].vel = (self.Peces[i].vel / self.Peces[i].vel.norma()) * self.__maxVel

		# Actualizo el centro de masa y la velovidad media
		self.CMyVM()
		
		self.iteracion += 1
		if self.iteracion % PLOTEA == 0:
			self.gif.append( [self.plotea()] )

	def plotea(self):
		X  = np.array( [self.Peces[i].pos.r[0] for i in range(self.N)] )
		Y  = np.array( [self.Peces[i].pos.r[1] for i in range(self.N)] )
		Vx = np.array( [self.Peces[i].vel.r[0] for i in range(self.N)] )
		Vy = np.array( [self.Peces[i].vel.r[1] for i in range(self.N)] )
		return plt.quiver(X,Y,Vx,Vy)

	def print(self):
		for i in range(self.N):
			print("Pez: {}".format(i), end=' ')
			print("pos = ({:.2f}, {:.2f})".format(self.Peces[i].pos.r[0], self.Peces[i].pos.r[1]), end=' ')
			print("vel = ({:.2f}, {:.2f})".format(self.Peces[i].vel.r[0], self.Peces[i].vel.r[1]))
		print("")


from matplotlib import animation

def ejercicio_13(dim,vMax,dMax):
	c = Cardumen()
	c.initialize(dim,vMax,dMax)

	for i in range(NUM_ITERAC):
		c.doStep()
		c.print()

	ani = animation.ArtistAnimation(c.fig, c.gif, interval=50, blit=True, repeat_delay=1000)

	plt.show()

	#ani.save("cardumen_1.mp4")

#---------------------------------
#			Ejercicio 14
#---------------------------------

def ejercicio_14():
	n = np.arange(10,61,5)
	coincidences = np.array([])

	for i in n:
		count = 0
		for _ in range(1000):
			bd = np.random.randint(1,366,i)		# Genero un array de i elementos con valores enteros entre 1-365
			c , r = np.unique(bd, return_counts=True)	# esta funcion devuelve los valore ordenados y cant repeticiones
			if r.size != i:
				count += 1
		coincidences = np.append(coincidences, (count/1000)*100)

	plt.title("Ejercicio 14")
	plt.xlabel("# de personas en el grupo", fontsize=14)
	plt.ylabel(r"$\%$ de cumpleaños repetido", fontsize=14)
	plt.plot(n, coincidences, 'bo')
	#plt.savefig('Informe/ej_14.svg', format='svg', bbox_inches='tight')
	plt.show()


#---------------------------------
#			Ejercicio 15
#---------------------------------

class Noiser(object):
	"""docstring for Noiser"""
	def __init__(self, minV, maxV):
		self.minV = minV
		self.maxV = maxV
	def __call__(self,x):
		return x + np.random.uniform(self.minV,self.maxV)
		
def ejercicio_15(minV,maxV):
	v_Noiser = np.vectorize(Noiser(minV,maxV))

	a = np.zeros(50)

	print("Ejercicio 15: minV = {} maxV = {}".format(minV,maxV))
	print(v_Noiser(a))





#---------------------------------
#
#---------------------------------

if __name__ == '__main__':

	ejercicio_1()

	ejercicio_2()

	ejercicio_3(1,1,1)

	ejercicio_4(-3,2,16)

	ejercicio_5(5,3,2)

	ejercicio_6(5,3,2)

	ejercicio_7()

	ejercicio_10()

	ejercicio_11()

	ejercicio_12()

	ejercicio_12_histograma()

	ejercicio_13(40,4,1)

	ejercicio_14()

	ejercicio_15(-1,1)