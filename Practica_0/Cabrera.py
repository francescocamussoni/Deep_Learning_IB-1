#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19-08-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt

#------------------------
#		Ejercicio 1
#------------------------

A = np.array([[ 1,  0,  1],
			  [ 2, -1,  1],
			  [-3,  2, -2]])

b = np.array([[-2],
			  [ 1],
			  [-1]])

invA = np.linalg.inv(A)

res = np.dot(invA,b)

print("res = {}".format(res))

#------------------------
#		Ejercicio 
#------------------------