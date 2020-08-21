#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_12.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://matplotlib.org/tutorials/colors/colormaps.html
"""

import os
import numpy as np
from matplotlib import pyplot as plt


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
plt.savefig('Informe/ej_12.pdf', format='pdf', bbox_inches='tight')

#plt.gca().get_xaxis().set_visible(False)

plt.show()
