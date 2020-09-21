#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: layers.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import myModules.activations as act
import myModules.regularizers as reg

class BaseLayer(object):
    def __init__(self):
        pass

    def get_output_shape(self):
        pass

    def set_output_shape(self):
        pass

class Input(BaseLayer):
    def __init__(self):
        #Completar
        pass

    def get_output_shape(self):
        #Completar
        pass

    def set_output_shape(self):
        #Completar
        pass

class ConcatInput(BaseLayer):
    def __init__(self):
        #Completar
        pass

    def __call__(self):
        #Completar
        pass

    def set_output_shape(self):
        #Completar
        pass

    def get_output_shape(self):
        #Completar
        pass

    def get_input1_shape(self):
        #Completar
        pass

    def get_input2_shape(self):
        #Completar
        pass

class WLayer(BaseLayer):
    def __init__(self, nraws, units, activation=act.Linear, regu=reg.L2, w = 1e-3):
        self.shape = [nraws, units]
        self.activation = activation
        self.reg = regu

        self.W = np.random.uniform(-w,w, size=(self.shape[0]+1 , self.shape[1]) )

    def get_input_shape(self):
        return self.shape[0]
    
    def get_output_shape(self):
        return self.shape[1]

    def set_input_shape(self):
        #Completar WTF
        pass

    def set_output_shape(self):
        ####### WTF
        pass

    def get_weights(self):
        return self.W

    def update_weights(self):
        # Completar
        pass


class Dense(WLayer):
    def __init__(self, nraws, units, activation=act.Linear, regu=reg.L2, w = 1e-3):
        super().__init__(nraws, units, activation, regu, w)
    
    def __call__(self,X):
        return self.activation(self.dot(X))
    
    def dot(self,X):
        X_p = np.hstack((np.ones((len(X),1)), X))
        return np.dot(X_p, self.W)
        

