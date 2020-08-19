#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_3.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt


def Bhaskara(a,b=0,c=0):
	return np.roots([a,b,c]) 	# No se si esto es trampa

if __name__ == '__main__':
	res = Bhaskara(1,1,1)

	print("Ejercicio 3: Las raices de x²+x+1 son:")
	print("x_1 = {}\nx_2 = {}".format(res[0], res[1]))