#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 21-09-2020
File: regularizers.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

class Ragularizer(object):
    def __init__(self, landa=1e-2):
        self.landa = landa

    def __call__(self):
        pass

    def gradient(self):
        pass

class L2(Ragularizer):

    def __call__(self, W):
        return self.landa * np.sum(W*W)
    
    def gradient(self,W):
        return 2*self.landa*W

class L1(Ragularizer):

    def __call__(self,W):
        return self.landa * np.sum(np.abs(W))
    
    def gradient(self,W):
        return self.landa*np.sign(W)