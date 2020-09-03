#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 29-08-2020
File: ej_2.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# Soy una semilla para que me cree los mismos puntos aleatorios
#np.random.seed(22)

N = 3   # Dimension
numData = 1000



 
#####################################################
# Primer cluster
mean1 = np.array([0,0])
c1 = np.random.normal(mean1,1, size=(numData,2))

#####################################################
# Segundo cluster
mean2 = np.array([3.5,0])
c2 = np.random.normal(mean2,1, size=(numData,2))

#####################################################
# Tercer cluster
mean3 = np.array([7,7])
c3 = np.random.normal(mean3,1, size=(numData,2))








data = np.concatenate((c1, c2, c3))

id1, id2, id3 = np.random.randint(0, 3 * numData, 3)     # Aca tengo la cantidad de clusteter

mean1 = data[id1]
mean2 = data[id2]
mean3 = data[id3]

mean = np.array([data[id1], data[id2], data[id3]])
old = np.array([[0,0], [0,0], [0,0]])   # tengo que mejorar esto


plt.scatter(c1[:,0], c1[:,1])
plt.scatter(c2[:,0], c2[:,1])
plt.scatter(c3[:,0], c3[:,1])
plt.scatter(mean[0][0], mean[0][1], c='k', marker='x')
plt.scatter(mean[1][0], mean[1][1], c='k', marker='x')
plt.scatter(mean[2][0], mean[2][1], c='k', marker='x')
plt.show()



print(mean)
print(old)


while( not (old == mean).all() ):
    old = np.copy(mean)
    matrizota = data - mean[:, np.newaxis]  # Esto tiene k matrices, c/u con las dist al resp centro

    #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    print(matrizota)

    dist = np.linalg.norm(matrizota, axis=2)

    indices = np.argmin(dist, axis=0)

    cluster1 = data[indices == 0]
    cluster2 = data[indices == 1]
    cluster3 = data[indices == 2]

    mean[0][0] = cluster1[:,0].mean()
    mean[0][1] = cluster1[:,1].mean()
    mean[1][0] = cluster2[:,0].mean()
    mean[1][1] = cluster2[:,1].mean()
    mean[2][0] = cluster3[:,0].mean()
    mean[2][1] = cluster3[:,1].mean()

    plt.scatter(cluster1[:,0], cluster1[:,1])
    plt.scatter(cluster2[:,0], cluster2[:,1])
    plt.scatter(cluster3[:,0], cluster3[:,1])
    plt.scatter(mean[0][0], mean[0][1], c='k', marker='x')
    plt.scatter(mean[1][0], mean[1][1], c='k', marker='x')
    plt.scatter(mean[2][0], mean[2][1], c='k', marker='x')

    plt.show()

    #print(mean)
    #print(old)
