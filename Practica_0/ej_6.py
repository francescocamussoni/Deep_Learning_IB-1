#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_6.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt


from ej_5 import Lineal

class Exponencial(Lineal):
	"""docstring for Exponencial"""
	def __init__(self, a,b):
		super(Exponencial, self).__init__(a,b)
	def __call__(self,x):
		return self.a * x**self.b

