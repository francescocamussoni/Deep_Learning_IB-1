#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 20-09-2020
File: metrics.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

def __yyZeros(y_true, shape_):
    zeros = np.zeros(shape = shape_)
    zeros[ np.arange(y_true.shape[0]), y_true] = 1
    return zeros

def mse(scores, y_true):
    yy = __yyZeros(y_true, scores.shape)
    return ((scores-yy)**2).sum(axis=1).mean()

def accuracy(scores, y_true):
    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
    y_predict = np.argmax(scores, axis=1)
    return (y_predict == y_true).mean()*100

def acc_XOR(scores, y_true):
	S = np.copy(scores)
	S[S>0.9]  =  1
	S[S<-0.9] = -1
	return (S == y_true).mean()*100
