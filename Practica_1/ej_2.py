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

import seaborn as snn

snn.set(font_scale = 1)


class createCluster():
    def __init__(self, N=3, p=4):
        self.N = N      # Dimension
        self.p = p      # Cantidad de clases
        self.numData = 100
    
    def checkDist(self, newMean):
        self.dist = np.linalg.norm(self.means - newMean, axis=1)
        #print(self.dist)

        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
        if( (self.dist>4).all() and (self.dist<6).any()):
            return True
        else:
            return False

    
    def createData(self):

        # El primer cluster lo pongo en el origen
        mean1 = np.zeros(self.N)
        c1 = np.random.normal(mean1,1, size=(self.numData,self.N))

        self.means = np.array([mean1])
        self.clusters = np.array([c1])

        # Segundo cluster, apenas solapado
        mean2 = np.zeros(self.N)
        mean2[0] = 3.5
        c2 = np.random.normal(mean2,1, size=(self.numData,self.N))


        self.means = np.append(self.means, [mean2], axis=0)
        self.clusters = np.append(self.clusters, [c2], axis=0)


        for _ in range(2,self.p):
            mean = np.random.uniform(-10,10, self.N)

            while(not self.checkDist(mean)):
                mean = np.random.uniform(-10,10, self.N)
            
            #c = np.random.normal(mean,1, size=(self.numData,self.N))
            c = np.random.normal(mean, np.random.uniform(0,3), size=(self.numData,self.N))
            
            self.means = np.append(self.means, [mean], axis=0)
            self.clusters = np.append(self.clusters, [c], axis=0)
        
        self.data = self.clusters.reshape( (self.p * self.numData, self.N) )
        
        return self.data
    
    def getClusters(self):
        return self.clusters

# Para armar el video
#fig = plt.figure()
#ims = []


class kMeans():
    def __init__(self, N=3, k=4, n_iter=100):
        self.N = N
        self.k = k
        self.numData = 100
        self.iter = n_iter
    
    def execute(self, data, plot=True):
        self.data = np.copy(data)

        # Tomo centros iniciales de forma aleatoria
        index = np.random.randint(0, len(self.data), self.k)
        self.means = self.data[index]

        self.old = None     # Aca voy a ir guardando los centros viejos
        count = 0           # Contador para cortar en caso de no converger

        #self.cluster = np.zeros(shape=(self.k, 1,2))      # Para guardar los puntos de c/cluster
        self.cluster = np.empty(self.k, dtype='object')      # Para guardar los puntos de c/cluster

        while( not (self.old == self.means).all() or (count == self.iter) ):
            self.old = np.copy(self.means)

            diff = data - self.means[:, np.newaxis]     # Esto tiene k arrays, c/u tiene la diferencias de
                                                        # TODOS los puntos al centro k-esimo 

            dist = np.linalg.norm(diff, axis=2)         # Calculo las distancias de cada puntos

            indices = np.argmin(dist, axis=0)           # De los k arrays, queremos ver, para cada punto,
                                                        # cual nos da menor distancia
                                                        
            for i in range(self.k):
                self.cluster[i] = self.data[indices == i]       # Actualizo los puntos del i-esimo cluster
                self.means[i] = self.cluster[i].mean(axis=0)    # Actualizo el centro
            
            
            if(plot):                                   # Ploteo solo las 2 primeras dimensiones
                # aux = []  # Para armar el video
                plt.gca().set_prop_cycle(None)
                for i in range(self.k):
                    plt.scatter(self.cluster[i][:,0], self.cluster[i][:,1], marker='.' )
                    plt.scatter(self.means[i][0], self.means[i][1], c='k', marker='x')
                    #aux.append( plt.scatter(self.cluster[i][:,0], self.cluster[i][:,1], marker='.' ))
                    #aux.append( plt.scatter(self.means[i][0], self.means[i][1], c='k', marker='x')  )
                #ims.append( (aux) )
                plt.title(r'p=4  k={}   iteracion = {}'.format(self.k,count))
                #plt.savefig('Informe/2/2_{}_{}.pdf'.format(self.k,count), format='pdf', bbox_inches='tight')
                plt.pause(1)
                #plt.close()
                plt.clf()
                #plt.show()
            
            count += 1






# Doy una semilla para que me cree los mismos puntos aleatorios
np.random.seed(6)


N = 3

kk = createCluster(N)
data = kk.createData()
cluster = kk.getClusters()
#aux = []       # Para armar el video

# Grafico la distribucion original
for i in range(len(cluster)):
    plt.scatter(cluster[i][:,0], cluster[i][:,1], marker='.')

#for _ in range(4):     # Para armar el video
#    ims.append((aux))  # Para armar el video

plt.title(r"Distribucion cluster con $p=4$")
#plt.savefig('Informe/2/2_Distribucion_Clusters.pdf', format='pdf', bbox_inches='tight')
plt.pause(3)
plt.clf()



np.random.seed(1)

# Ejecuto el algoritmo para k desde 2 a 5
for i in range(2,6):
    knn = kMeans(N,k=i)
    knn.execute(data)


# De aca para abajo es para armar el video, asi que lo comento
"""
import matplotlib.animation as animation


im_ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=3000,
                                   blit=True)

plt.show()

im_ani.save('Intento.mp4', metadata={'artist':'FMC'})
"""