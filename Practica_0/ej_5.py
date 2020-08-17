#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_5.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt



class Lineal(object):
	"""docstring for Lineal"""
	def __init__(self, a,b):
		#super(Lineal, self).__init__()
		self.a = a
		self.b = b
	def __call__(self,x):
		return self.a*x+self.b
		