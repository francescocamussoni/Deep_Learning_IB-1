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


class R2(object):
	"""docstring for R2"""
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
	"""docstring for Pez"""
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
		self.N = None
		self.__maxVel = None
		self.__maxDist = None
		self.Peces = np.array([])
		self.CM = None 			# Centro del cardumen
		self.VM = None			# Velocidad media del cardumen

	def CMyVM(self):
		aux = self.Peces.sum()
		print(type(aux))
		self.CM = aux.pos / self.N
		self.VM = aux.vel / self.N

	def initialize(self, N, maxVel, maxDist):
		self.N = N
		self.__maxVel = maxVel
		self.__maxDist = maxDist
		for i in range(N):
			v = random.uniform(0,maxVel)
			t = 2*np.pi*random.random()
			pez = Pez(R2(random.uniform(0,40),random.uniform(0,40)) , R2(v*np.cos(t), v*np.sin(t)))
			self.Peces = np.append(self.Peces, pez)
		self.CMyVM()
		#self.CM = self.CM()
		#self.VM = self.VM()

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

	def rebote(self,Pez):
		pass

	def doStep(self):
		# Calculo los delta de velocidad de cada pez
		DeltaV = np.array([self.reglas(i) for i in range(self.N)])
		# Modifico las posiciones y velocidades
		for i in range(self.N):
			self.Peces[i].pos = self.Peces[i].pos + self.Peces[i].vel
			self.Peces[i].vel = self.Peces[i].vel + DeltaV[i]
			if self.Peces[i].pos.r[0] > 40:
				self.Peces[i].pos.r[0] = 80 - self.Peces[i].pos.r[0]	# Lo hago rebotar en x
				self.Peces[i].vel.r[0] = -self.Peces[i].vel.r[0]		# Lo hago rebotar
			elif self.Peces[i].pos.r[0] < 0:
				self.Peces[i].pos.r[0] = - self.Peces[i].pos.r[0]
				self.Peces[i].vel.r[0] = -self.Peces[i].vel.r[0]		# Lo hago rebotar

			if self.Peces[i].pos.r[1] > 40:
				self.Peces[i].pos.r[1] = 80 - self.Peces[i].pos.r[1]	# Lo hago rebotar en x
				self.Peces[i].vel.r[1] = -self.Peces[i].vel.r[1]		# Lo hago rebotar
			elif self.Peces[i].pos.r[1] < 0:
				self.Peces[i].pos.r[1] = - self.Peces[i].pos.r[1]
				self.Peces[i].vel.r[1] = -self.Peces[i].vel.r[1]		# Lo hago rebotar
			# Limito la velocidad
			if self.Peces[i].vel.norma() > self.__maxVel:
				self.Peces[i].vel = (self.Peces[i].vel / self.Peces[i].vel.norma()) * self.__maxVel

	def print(self):
		for i in range(self.N):
			print("Pez: {}".format(i), end=' ')
			print("pos = ({}, {})".format(self.Peces[i].pos.r[0], self.Peces[i].pos.r[1]), end=' ')
			print("vel = ({}, {})".format(self.Peces[i].vel.r[0], self.Peces[i].vel.r[1]))





		
		
c = Cardumen()
c.initialize(5, 10, 5)
for i in range(5):
	c.doStep()
	c.print()