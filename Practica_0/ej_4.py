#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_4.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from ej_3 import Bhaskara

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
	#plt.title("Ejercicio 4")
	plt.savefig('Informe/ej_4.pdf', format='pdf', bbox_inches='tight')
	plt.show()

grafYRaices(-3,2,16)