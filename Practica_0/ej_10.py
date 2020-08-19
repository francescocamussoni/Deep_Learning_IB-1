#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_10.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html
"""

import os
import numpy as np
from matplotlib import pyplot as plt


def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)
X, Y = np.meshgrid(x,y)

plt.axes([0.025, 0.025, 0.95, 0.95])
plt.imshow(f(X, Y), cmap='bone', origin='lower')	# De los tipos de interpolacion, el unico que deja pixeleada la imagen
													# es "nearest", pero no cambia nada con dejarlo en blanco (creo)
plt.colorbar(shrink=.8)

plt.xticks(())
plt.yticks(())
plt.savefig('Informe/ej_10.pdf', format='pdf', bbox_inches='tight')
plt.show()

