#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16-08-2020
File: ej_2.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
				https://en.wikipedia.org/wiki/Gamma_distribution
"""

import numpy as np
import matplotlib.pyplot as plt

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
ax.set_ylabel("Repeticiones", fontsize=15)
fig.savefig('Informe/ej_2.pdf', format='pdf', bbox_inches='tight')
plt.show()