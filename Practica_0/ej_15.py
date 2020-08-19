#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19-08-2020
File: ej_15.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt

class Noiser(object):
	"""docstring for Noiser"""
	def __init__(self, minV, maxV):
		self.minV = minV
		self.maxV = maxV
		#self.__call__ = np.vectorize(self.f)
	#def f(self,x):
	#	return x + np.random.uniform(self.minV,self.maxV)
		#self.
	def __call__(self,x):
		return x + np.random.uniform(self.minV,self.maxV)
		

v_Noiser = np.vectorize(Noiser(0,2))
