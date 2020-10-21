#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 29-08-2020
File: ej_1.py
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

# Doy una semilla para que me cree los mismos puntos aleatorios
np.random.seed(10)


class ajusteLineal():
    def __init__(self,N, argMin=0, argMax=10):
        self.N = N
        self.min = argMin
        self.max = argMax
        # Para no complicarme, los coeficientes del hiperplano los hago entero en un rango razonable
        self.coef = np.random.uniform(-10,10,(N+1,1))         

    def randomData(self, M, std=1):
        self.X = np.random.uniform(self.min, self.max, size=(M,self.N))
        self.X = np.hstack((np.ones((M,1)), self.X))          # Agrego 1 para el bias
        self.Y = self.X @ self.coef                           # Estos son los y reales
        noise = np.random.normal(0, std, size=self.Y.shape)
        self.Y_noise = self.Y + noise
        # Aca saco los coeficientes ajustados
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y_noise

    def plotAll(self, save=True):
        fig = plt.figure(figsize=(10,5))
        ax = plt.subplot(111)
        
        for i in range(self.N):
            ax.plot(self.X[:,i+1], self.Y_noise - np.delete(self.X,i+1,axis=1) @ np.delete(self.coef,i+1,axis=0) + self.beta[0], 'o', label='Dim. {}'.format(i+1)) # A esta es la que le tengo que restar
        for i in range(self.N-1):
            ax.plot(self.X[:,i+1], self.X[:,i+1] * self.beta[i+1] + self.beta[0], 'k')
        ax.plot(self.X[:,self.N], self.X[:,self.N] * self.beta[self.N] + self.beta[0], 'k', label="Ajuste")
    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax.set_xlabel("X", fontsize=15)
        ax.set_ylabel("Y", fontsize=15)
        if(save):
            plt.savefig('Informe/1_Todos.pdf', format='pdf', bbox_inches='tight')
        plt.show()
    
    def plotDim(self,i, save=True):
        fig = plt.figure()
        ax = plt.subplot(111)

        ax.plot(self.X[:,i], self.Y_noise - np.delete(self.X,i,axis=1) @ np.delete(self.coef,i,axis=0) + self.beta[0], 'or', label='Dim. {}'.format(i))
        ax.plot(self.X[:,i], self.X[:,i] * self.beta[i] + self.beta[0], '--k', label='Ajuste {}'.format(i))
        ax.legend(loc="best")
        ax.set_xlabel("X", fontsize=15)
        ax.set_ylabel("Y", fontsize=15)
        if(save):
            plt.savefig('Informe/1_UnaDimension_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
        plt.show()
    
    def getError(self):
        diff = self.coef - self.beta
        return np.linalg.norm(diff[1:]) / (np.sqrt(self.N))


def barridoDatos(save=True):
    fig = plt.figure()
    ax = plt.subplot(111)

    numDatos = 50

    for dim in range(10,numDatos,10):
        log_error = []
        fit= ajusteLineal(dim)
        for i in range(dim,numDatos):
            error = 0
            for _ in range(100):
                fit.randomData(i)
                error += fit.getError()
            log_error += [error/100.0]
        
        datos = np.arange(dim,numDatos)
        log_error = np.array(log_error)
        
        ax.axvline(dim, color="gray", linestyle="--", )
        ax.semilogy(datos, log_error, 'o', label='Dim. {}'.format(dim))


    ax.legend(loc='best')
    ax.set_xlabel("Cantidad de datos", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)

    if(save):    
        plt.savefig('Informe/1_Barrido_en_Datos.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def barridoDim(save=True):
    fig = plt.figure(figsize=(7.4,4.8))
    ax = plt.subplot(111)

    for nDatos in range(40, 9, -10):
        log_error = []
        for dim in range(1,nDatos+1):
            fit = ajusteLineal(dim)
            error = 0
            for _ in range(100):
                fit.randomData(nDatos)
                error += fit.getError()
            log_error += [error/100.0]
        
        dimension = np.arange(1,nDatos+1)
        log_error = np.array(log_error)

        ax.axvline(nDatos, color="gray", linestyle="--")
        ax.semilogy(dimension, log_error, 'o', label='N° Datos = {}'.format(nDatos))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])        
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel("Dimension", fontsize=15)
    ax.set_ylabel("Error", fontsize=15)

    
    if(save):
        plt.savefig('Informe/1_Barido_en_Dimension.pdf', format='pdf', bbox_inches='tight')
    plt.show()







kk = ajusteLineal(5)
kk.randomData(100,std=5)
kk.plotAll()
kk.plotDim(3)

barridoDatos()

barridoDim()

