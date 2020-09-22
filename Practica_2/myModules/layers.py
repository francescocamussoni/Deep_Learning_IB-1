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
    def __init__(self,x_shape):
        self.out_shape = x_shape
    
    def __call__(self,X):
        return X

    def get_output_shape(self):
        return self.out_shape

    def set_output_shape(self):
        # XXX WTF
        pass

    

class ConcatInput(BaseLayer):
    def __init__(self, layerX):
        # XXX Pasar layer o el numero de columnas?
        self.X_shape = layerX.get_output_shape()
        # self.S_shape = layerS.get_output_shape()

    def __call__(self,X,S):
        return np.hstack( (S,X) )       # X es el reinyectado

    def get_output_shape(self):
        return self.out_shape

    def get_input1_shape(self):
        return self.X_shape

    def get_input2_shape(self):
        return self.S_shape
    
    def set_input_shape(self,S_shape):
        self.S_shape = S_shape
        self.out_shape = self.X_shape + self.S_shape

    def set_output_shape(self):
        # XXX WTF
        pass

    

class Concat(BaseLayer):
    def __init__(self, layerS2, forward, iS2 = 0):
    # def __init__(self, layerS1, layerS2, forward, iS2 = 0):
        # self.S1_shape = 0
        # self.S1_shape = layerS1.get_output_shape()
        self.S2_shape = layerS2.get_output_shape()
        self.forward = forward
        self.i = iS2
    
    def __call__(self,X,S1):
        S2 = self.forward(X,self.i+1)           # XXX Tenemos dudas del +1
        return np.hstack( (S1,S2) )             # S2 es el reinyectado
    
    # def __call__(self,X):
    #     S = self.forward(self.i)
    #     return np.hstack( (X,S) )
    
    def get_output_shape(self):
        return self.out_shape
    
    def get_input1_shape(self):
        return self.S1_shape
    
    def get_input2_shape(self):
        return self.S2_shape
    
    def set_output_shape(self):
        # XXX WTF
        pass

    def set_input_shape(self,S1_shape):
        self.S1_shape = S1_shape
        self.out_shape = self.S1_shape + self.S2_shape
    




class WLayer(BaseLayer):
    def __init__(self, units, activation=act.Linear(), regu=reg.L2(), w = 1e-3, nraws=None):
        # XXX Se puede pasar el layer en vez del numero de columnas y hacer un get_shape
        self.in_shape = nraws
        self.out_shape = units
        self.activation = activation
        self.reg = regu
        self.w = w

    def __initW(self):
        self.W = np.random.uniform(-self.w,self.w, size=(self.in_shape+1 , self.out_shape) )

    def get_input_shape(self):
        return self.in_shape
    
    def get_output_shape(self):
        return self.out_shape

    def set_input_shape(self,in_shape):
        self.in_shape = in_shape
        self.__initW()

    def set_output_shape(self):
        ####### WTF
        pass

    def get_weights(self):
        return self.W

    def update_weights(self, dW):
        self.W -= dW

class Dense(WLayer):
    def __init__(self, units, activation=act.Linear(), regu=reg.L2(), w = 1e-3, nraws=None):
        super().__init__(units, activation, regu, w, nraws)
    
    def __call__(self,X):
        return self.activation(self.dot(X))
    
    def dot(self,X):
        X_p = np.hstack((np.ones((len(X),1)), X))
        return np.dot(X_p, self.W)
        

