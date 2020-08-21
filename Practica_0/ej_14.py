#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18-08-2020
File: eje_14.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
"""

import os
import numpy as np
from matplotlib import pyplot as plt


n = np.arange(10,61,5)
#coincidences = np.zeros(n.shape)
coincidences = np.array([])

for i in n:
	count = 0
	for _ in range(1000):
		bd = np.random.randint(1,366,i)		# Genero un array de i elementos con valores enteros entre 1-365
		c , r = np.unique(bd, return_counts=True)	# esta funcion devuelve los valore ordenados y cant repeticiones
		if r.size != i:
			count += 1
	coincidences = np.append(coincidences, (count/1000)*100)

#plt.title("Ejercicio 14")
plt.xlabel("# de personas en el grupo")
plt.ylabel(r"$\%$")
plt.plot(n, coincidences, 'bo')
plt.savefig('Informe/ej_14.pdf', format='pdf', bbox_inches='tight')
plt.show()