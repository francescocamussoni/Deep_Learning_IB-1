#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: optimizers.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

class Optimizer(object):
    def __init__(self, lr=1e-3):
        self.lr = lr
    
    def __call__(self, X, Y, model):
        pass

    def update_weights(self, W, gradW):
        pass


class SGD(Optimizer):
    def __init__(self, lr=1e-3):
        super().__init__(lr)
        # self.bs = bs
    
    def __call__(self, X, Y, model,bs=False):
        if(not bs):
            # No hacer los bacths ndafeabu
            model.backward(X,Y)
        else:
            n_bacht = int(len(X)/bs)

            idx = np.arange(len(X))
            np.random.shuffle(idx)

            for i in range(n_bacht):

                bIdx = idx[bs*i: bs*(i+1)]

                x_bacht = X[bIdx]
                y_bacht = Y[bIdx]

                model.backward(x_bacht,y_bacht)
    
    def update_weights(self, layer, gradW):
        dW = self.lr * ( gradW + layer.reg.gradient(layer.get_weights()) )
        return dW
