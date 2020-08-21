#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_13.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import random

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

		if self.iteracion % (NUM_ITERAC // 5) == 1:

			X  = np.array( [self.Peces[i].pos.r[0] for i in range(self.N)] )
			Y  = np.array( [self.Peces[i].pos.r[1] for i in range(self.N)] )
			Vx = np.array( [self.Peces[i].vel.r[0] for i in range(self.N)] )
			Vy = np.array( [self.Peces[i].vel.r[1] for i in range(self.N)] )

			fig = plt.figure()

			plt.xlim(0,self.dim)
			plt.ylim(0,self.dim)
			plt.xticks(())
			plt.yticks(())	

			ax = fig.add_subplot(1, 1, 1)
			ax = plt.quiver(X,Y,Vx,Vy)
			plt.savefig('Informe/ej_13_{}.pdf'.format(self.iteracion), format='pdf', bbox_inches='tight')


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
		
c = Cardumen()
c.initialize(40, 4, 1)

for i in range(NUM_ITERAC):
	c.doStep()
	c.print()

ani = animation.ArtistAnimation(c.fig, c.gif, interval=50, blit=True, repeat_delay=1000)

plt.show()

#ani.save("cardumen_1.mp4")
