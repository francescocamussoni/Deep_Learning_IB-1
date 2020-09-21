#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: activations.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

class Activation(object):
    def __call__(self):
        pass

    def gradient(self):
        pass

class Relu(Activation):
    def __call__(self, x_):
        return np.maximum(0,x_)
    
    def gradient(self, x_):
        return np.heaviside(x_, 0)

class Sigmoid(Activation):
    def __call__(self, x_):
        return 1/(1 + np.exp(-x_))
    
    def gradient(self, x_):
        sigma = self.__call__(x_)
        return (1-sigma)*sigma

class Linear(Activation):
    def __call__(self, x_):
        return x_
    
    def gradient(self, x_):
        return 1

class Tanh(Activation):
    def __call__(self, x_):
        return np.tanh(x_)
    
    def gradient(self, x_):
        return 1 - np.tanh(x_)**2

class LeakyRelu(Activation):
    def __call__(self, x_):
        return np.maximum(0.1*x_, x_)
    
    def gradient(self, x_):
        return np.heaviside(x_, 0) + np.heaviside(-x_, 0)*0.1
