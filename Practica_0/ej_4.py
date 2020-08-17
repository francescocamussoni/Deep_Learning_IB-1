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

def grafYRaices(a,b,c):
	x = np.linspace(-4.5, 4, 1000)
	p = np.polyval([a,b,c], x)
	r1, r2 = Bhaskara(a,b,c)

	plt.plot(x,p,'b-')
	if r1.imag == 0 and r1 < 4 and r1 > -4.5:
		plt.plot(r1,0,'or')
		plt.annotate(s=r"$x_1=${:.2f}".format(r1), xy=(r1, 0), xytext=(r1+0.3, 0))
	if r2.imag == 0 and r2 < 4 and r2 > -4.5:
		plt.plot(r2,0,'or')
		plt.annotate(s=r"$x_2=${:.2f}".format(r2), xy=(r2, 0), xytext=(r2+0.3, 0))
	plt.show()