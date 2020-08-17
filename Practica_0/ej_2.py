#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16-08-2020
File: ej_2.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma
import scipy.special as sps

shape = 2	# Se suele notar con k
scale = 3	# Se suele notar con $\theta$


data = np.random.gamma(shape,scale,1000)

mean = data.mean()		# mu = k * $\theta$
std = data.std()		# sigma² = k * $\theta^2$

x = np.linspace(0, data.max(), 1000)

k = (mean / std)**2
theta = (std**2) / mean

dist = gamma(a=k, scale = theta)

plt.hist(data, bins=30)
plt.plot(x, dist.pdf(x) *1000)
plt.show()