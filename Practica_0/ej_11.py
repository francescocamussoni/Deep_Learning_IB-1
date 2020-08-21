#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_11.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt


def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x,y)

plt.figure()
plt.axes([0.025, 0.025, 0.95, 0.95])
plt.contourf(X, Y, f(X, Y), 7, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y, f(X, Y), 7, colors='black')
plt.clabel(C, inline=5, fontsize=8)

plt.xticks(())
plt.yticks(())
plt.savefig('Informe/ej_11.pdf', format='pdf', bbox_inches='tight')
plt.show()