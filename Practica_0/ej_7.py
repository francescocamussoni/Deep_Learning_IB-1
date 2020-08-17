#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_7.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
from matplotlib import pyplot as plt


import circunferencia as circle

print("Valor de PI importando con el alias \"circle\": {}".format(circle.PI))
print("Valor de del area de un circulo de radio 2, importando con el alias \"circle\": {}".format(circle.area(2)))

from circunferencia import PI, area

print("Valor de PI importando directamente: {}".format(PI))
print("Valor de del area de un circulo de radio 2, importando directamente: {}".format(area(2)))

if PI is circle.PI and area is circle.area:
	print("Son el mismo objeto")
else:
	print("Son objetos distintos")