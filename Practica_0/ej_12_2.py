#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17-08-2020
File: ej_12.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://matplotlib.org/tutorials/colors/colormaps.html
"""

import os
import numpy as np
from matplotlib import pyplot as plt


n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y, X)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    #ax.tick_params(axis="x", labelbottom=False)
    #ax.tick_params(axis="y", labelleft=False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax_histx.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)


    #ax_histx.tick_params(axis="x", labelbottom=False)
    #ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(X, Y, c=T, cmap='jet', alpha=0.5, edgecolors='gray')  	# https://matplotlib.org/tutorials/colors/colormaps.html

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(X)), np.max(np.abs(Y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(X, bins=bins)
    ax_histy.hist(Y, bins=bins, orientation='horizontal')

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# use the previously defined function
scatter_hist(X, Y, ax, ax_histx, ax_histy)

plt.savefig('Informe/ej_12_Histograma.pdf', format='pdf', bbox_inches='tight')

plt.show()
