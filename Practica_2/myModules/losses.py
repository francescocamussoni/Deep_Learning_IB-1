#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: losses.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

class Loss(object):
    def __call__(self):
        pass

    def gradient(self):
        pass

class MSE(Loss):
    def __yyZeros(self, y_true, shape_):
        zeros = np.zeros(shape = shape_)
        zeros[np.arange(y_true.shape[0]), y_true] = 1
        return zeros
    
    def __call__(self, scores, y_true):
        yy = self.__yyZeros(y_true, scores.shape)
        return ((scores-yy)**2).sum(axis=1).mean()
    
    def gradient(self, scores, y_true):
        yy = self.__yyZeros(y_true, scores.shape)
        return 2*(scores-yy)/len(y_true)

class MSE_XOR(Loss):
    def __call__(self, scores, y_true):
        return ((scores-y_true)**2).mean()
    
    def gradient(self, scores, y_true):
        return 2*(scores-y_true)/len(y_true)

class CCE(Loss):
    def __call__(self, scores, y_true):
        s_r = scores - scores.max(axis=1)[:,np.newaxis]

        y_idx = np.arange(y_true.size)
        y_win = s_r[y_idx, y_true]

        exp = np.exp(s_r)

        sumatoria = exp.sum(axis=1)

        log_softmax = np.log(sumatoria) - y_win

        return log_softmax.mean()
    
    def gradient(self, scores, y_true):
        s_r = scores - scores.max(axis=1)[:,np.newaxis]

        y_idx = np.arange(y_true.size)

        exp = np.exp(s_r)

        sumatoria = exp.sum(axis=1)

        softmax_fun = (1.0/sumatoria)[:,np.newaxis] * exp
        softmax_fun[y_idx, y_true] -= 1

        return softmax_fun / len(y_true)









