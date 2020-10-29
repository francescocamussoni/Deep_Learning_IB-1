#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 28-10-2020
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras import layers, activations, optimizers
from tensorflow.keras import regularizers


small_dataset = True
path_data = "/home/cabre1994/Desktop/Deep_Learning/Deep_Learning_IB/Datasets"

if small_dataset:
    path_data = os.path.join(path_data, "dogs-vs-cats_small")
else:
    path_data = os.path.join(path_data, "dogs-vs-cats")

