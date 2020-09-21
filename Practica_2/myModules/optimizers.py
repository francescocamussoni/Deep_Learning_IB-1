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
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, X, Y, model):
        pass

    def update_weights(self, W, gradW):
        pass


class SGD(Optimizer):
    def __init__(self, lr, bs=False):
        super().__init__(lr)
        self.bs = bs
    
    def __call__(self, X, Y, model):
        if(not self.bs):
            # No hacer los bacths ndafeabu
            model.backward(X,Y)
        else:
            # Aca si lo tengo que separar
            model.backward(X,Y)
    
    def update_weights(self, W, gradW):
        pass
