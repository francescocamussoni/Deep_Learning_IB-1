#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 04-09-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from keras.datasets import cifar10
from keras.datasets import mnist

import seaborn as snn

snn.set(font_scale = 1)


#---------------------------------
#			Ejercicio 1
#---------------------------------

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


def ejercicio_1():

    # Doy una semilla para que me cree los mismos puntos aleatorios

    np.random.seed(10)

    kk = ajusteLineal(5)
    kk.randomData(100,std=5)
    kk.plotAll(save=False)
    kk.plotDim(3,save=False)

    barridoDatos(save=False)

    barridoDim(save=False)




#---------------------------------
#			Ejercicio 2
#---------------------------------


"""
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
    
    def uglyData(self):
        # Voy a generarme dos cluster que sean un anillo y un circulo dentro del anillo
        # lo voy a hacer asi nomas porque son las 4am 

        numData = 400

        c1 = np.array([])
        c2 = []

        for _ in range(numData):
            p = np.random.normal((0,0), 1)
            d = np.linalg.norm(p)
            while(d > 2):
                p = np.random.normal((0,0), 1)
                d = np.linalg.norm(p)
            c1 = np.append(c1,p)
        
        c1 = c1.reshape((numData,2))
        
        for _ in range(numData):
            p = np.random.normal((0,0),4)
            d = np.linalg.norm(p)
            while(d<2 or d>3):
                p = np.random.normal((0,0), 4)
                d = np.linalg.norm(p)
            c2 = np.append(c2,p)
        
        c2 = c2.reshape((numData,2))

        self.dataFeo = np.append(c1,c2).reshape((2*numData,2))

        self.clusterFeo = self.dataFeo.reshape((2,numData,2))
        
        return self.dataFeo
    
    def uglyCluster(self):
        return self.clusterFeo
"""

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
    
    def uglyData(self):
        # Voy a generarme dos cluster que sean un anillo y un circulo dentro del anillo
        # lo voy a hacer asi nomas porque son las 4am 

        numData = 400

        c1 = np.array([])
        c2 = []

        for _ in range(numData):
            p = np.random.normal((0,0), 1)
            d = np.linalg.norm(p)
            while(d > 2):
                p = np.random.normal((0,0), 1)
                d = np.linalg.norm(p)
            c1 = np.append(c1,p)
        
        c1 = c1.reshape((numData,2))
        
        for _ in range(numData):
            p = np.random.normal((0,0),4)
            d = np.linalg.norm(p)
            while(d<2 or d>3):
                p = np.random.normal((0,0), 4)
                d = np.linalg.norm(p)
            c2 = np.append(c2,p)
        
        c2 = c2.reshape((numData,2))

        self.dataFeo = np.append(c1,c2).reshape((2*numData,2))

        self.clusterFeo = self.dataFeo.reshape((2,numData,2))

        ceros = np.zeros(numData)
        unos = np.ones(numData)

        self.labels = np.concatenate((ceros, unos))
        
        return self.dataFeo, self.labels
    
    def uglyCluster(self):
        return self.clusterFeo











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
                #plt.title(r'p=4  k={}   iteracion = {}'.format(self.k,count))
                #plt.savefig('Informe/2/2_{}_{}.pdf'.format(self.k,count), format='pdf', bbox_inches='tight')
                plt.xticks([])
                plt.yticks([])
                plt.pause(1)
                #plt.close()
                plt.clf()
                #plt.show()
            
            count += 1


def ejercicio_2():

    # Doy una semilla para que me cree los mismos puntos aleatorios
    np.random.seed(6)

    N = 3

    kk = createCluster(N)
    data = kk.createData()
    cluster = kk.getClusters()
    #aux = []       # Para armar el video


    for i in range(len(cluster)):                           # Grafico la distribucion original
        plt.scatter(cluster[i][:,0], cluster[i][:,1], marker='.')

    #for _ in range(4):     # Para armar el video
    #    ims.append((aux))  # Para armar el video

    plt.title(r"Distribucion cluster con $p=4$")
    #plt.savefig('Informe/2/2_Distribucion_Clusters.pdf', format='pdf', bbox_inches='tight')
    plt.pause(3)
    plt.clf()


    np.random.seed(1)

    for i in range(2,7):                        # Ejecuto el algoritmo para k desde 2 a 5
        knn = kMeans(N,k=i)
        knn.execute(data)



    ##########################
    # Ahora con una distribucion que no funciona
    ##########################

    kk2 = createCluster(2)
    dataFea, _ = kk2.uglyData()
    clusterFeo = kk2.uglyCluster()

    plt.scatter(clusterFeo[0][:,0], clusterFeo[0][:,1], marker='.', color='red')
    plt.scatter(clusterFeo[1][:,0], clusterFeo[1][:,1], marker='.', color='blue')
    #plt.savefig('Informe/2/2_FEO.pdf', format='pdf', bbox_inches='tight')
    plt.pause(3)
    plt.clf()
    #plt.show()
    plt.close()

    knn2 = kMeans(N=2,k=2)
    knn2.execute(dataFea)
    plt.show()



#---------------------------------
#			Ejercicio 3
#---------------------------------


class KNN():

    def __init__(self, k=1, orden=2,cast_float=False):
        self.k = k
        self.orden = orden
        self.cast_float = cast_float
    
    def norma(self,x1, x2, o):
        return np.linalg.norm(x1 - x2, ord=o)
    
    def saveData(self, dataX, dataY):
        if(self.cast_float):
            self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.float)
        else:
            self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.int16)
        self.Y = dataY.reshape(dataY.size)
    
    def predict(self,X):

        if(self.cast_float):
            X = X.reshape(len(X), X[0].size).astype(np.float)
        else:
            X = X.reshape(len(X), X[0].size).astype(np.int16)
        #X = X.reshape(len(X), X[0].size).astype(np.int16)

        y_pred = np.array([])
        # y_pred = [self._predict(x) for x in X]        # Hago el for en vez de esto xq quiero ver cuanto avanza
       
        for i in range(len(X)):
            if (((i % 1000) == 0) and len(X) >= 500):
                print(i, "/", len(X))
            y_pred = np.append(y_pred, self.predictOne(X[i]) )

        #return np.array(y_pred)
        return y_pred
    
    def predictOne(self,x):
        distances = [ self.norma(x, datax, self.orden) for datax in self.X]

        index = np.argsort(distances)[:self.k]          # Me quedo con los k indices con menor distancia

        labels = np.array([self.Y[i] for i in index])    # Y me fijo de que clase son

        #return np.bincount(labels).argmax()             # No se que hace si empata, yo solo confio en Numpy
        most_common = Counter(labels).most_common(1)     # Busco el mas repetido
        return most_common[0][0]



def accuracy(y_true, y_pred):
    y_true = y_true.reshape(y_true.size)
    y_pred = y_pred.reshape(y_pred.size)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy*100


def ejercicio_3_20test(c='mnist',k=1,norma=2):
    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    knn = KNN(k,norma)

    knn.saveData(x_train, y_train)

    predict = knn.predict(x_test[:20])

    print(c," - Precision para norma", norma, "y k =", k,
                 " para los primeros 20 test:", accuracy(predict, y_test[:20]) )


def ejercicio_3_Alltest(c='mnist',k=1,norma=2):
    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    knn = KNN(k,norma)

    knn.saveData(x_train, y_train)

    predict = knn.predict(x_test)

    print(c," - Precision para norma", norma, "y k =", k,
                 " para todos los test:", accuracy(predict, y_test) )



#---------------------------------
#			Ejercicio 4
#---------------------------------

"""
class KNN():

    def __init__(self, k=1, orden=2):
        self.k = k
        self.orden = orden
    
    def norma(self,x1, x2, o):
        return np.linalg.norm(x1 - x2, ord=o)
    
    def saveData(self, dataX, dataY):
        self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.float)
        self.Y = dataY.reshape(dataY.size)
    
    def predict(self,X):
        X = X.reshape(len(X), X[0].size).astype(np.float)
        y_pred = np.array([])
        # y_pred = [self._predict(x) for x in X]        # Hago el for en vez de esto xq quiero ver cuanto avanza
        for i in range(len(X)):
            if (((i % 1000) == 0) and len(X) >= 500):
                print(i, "/", len(X))
            y_pred = np.append(y_pred, self.predictOne(X[i]) )
        #return np.array(y_pred)

        return y_pred
    
    def predictOne(self,x):

        distances = [ self.norma(x, datax, self.orden) for datax in self.X]

        index = np.argsort(distances)[:self.k]          # Me quedo con los k indices con menor distancia

        labels = np.array([self.Y[i] for i in index])    # Y me fijo de que clase son

        #return np.bincount(labels).argmax()             # No se que hace si empata, yo solo confio en Numpy
        most_common = Counter(labels).most_common(1)     # Busco el mas repetido
        return most_common[0][0]
"""

class clusterEj4():
    def __init__(self):
        self.N = 2
        self.numData = 100
        pass

    def checkDist(self, newMean):
        self.dist = np.linalg.norm(self.means - newMean, axis=1)
        #print(self.dist)

        #import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT
        if( (self.dist>4).all() and (self.dist<6).any()):
            return True
        else:
            return False
    
    def nClusters(self,n):

        # El primer cluster lo pongo en el origen
        mean1 = np.zeros(self.N)
        c1 = np.random.normal(mean1,1, size=(self.numData,self.N))
        test1 = np.random.normal(mean1,1, size=(20,self.N))
        label1 = np.full(self.numData,0)
        t_label1 = np.full(20,0)

        self.means = np.array([mean1])
        
        self.clusters = np.array([c1])
        self.labels = np.array([label1])

        self.tests = np.array([test1])
        self.t_labels = np.array([t_label1])

        for i in range(1,n):
            mean = np.random.uniform(-10,10, self.N)
            std = np.random.uniform(0,2)

            while(not self.checkDist(mean)):
                mean = np.random.uniform(-10,10, self.N)
            
            #c = np.random.normal(mean,1, size=(self.numData,self.N))
            c = np.random.normal(mean, std, size=(self.numData,self.N))
            label = np.full(self.numData,i)

            test = np.random.normal(mean,std, size=(20,self.N))
            t_label = np.full(20,i)
            
            self.means = np.append(self.means, [mean], axis=0)
            self.clusters = np.append(self.clusters, [c], axis=0)
            self.labels = np.append(self.labels, [label], axis=0)
            self.tests = np.append(self.tests, [test], axis=0)
            self.t_labels = np.append(self.t_labels, [t_label], axis=0)
        
        return self.clusters, self.labels, self.tests, self.t_labels


def ejercicio4(n=7,k=1):

    np.random.seed(19)
        
    ej4 = clusterEj4()
    x_train, y_train, x_test, y_test = ej4.nClusters(n)

    #for i in range(n):
    #    plt.scatter(x_train[i][:,0], x_train[i][:,1,], marker='.')

    #plt.pause(3)
    #plt.close()
    #plt.show()

    knn = KNN(k=k,cast_float=True)

    x_train = x_train.reshape(n*100,2)
    x_test = x_test.reshape(n*20,2)

    knn.saveData(x_train, y_train)

    result = knn.predict(x_test)

    print("Precision para norma 2 y k = ",k," : ", accuracy(result, y_test) )

    #################################
    # Grafico
    #################################

    #steps = 200
    steps = 100

    xmin, xmax = x_train[:,0].min() - 1, x_train[:,0].max() + 1
    ymin, ymax = x_train[:,1].min() - 1, x_train[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)


    labels = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    z = labels.reshape(xx.shape)

    plt.figure()

    plt.contourf(xx, yy, z, n, alpha=.4, cmap='gist_rainbow')
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='gist_rainbow', edgecolors='gray', marker='.', alpha=0.7)
    
    plt.xticks(())
    plt.yticks(())
    
    #plt.savefig('Informe/2/4_k={}.pdf'.format(k), format='pdf', bbox_inches='tight')

    #plt.close(5)
    plt.show()


def ejercicio4_extra(k=1):

    kk2 = createCluster(2)
    train, label = kk2.uglyData()
    #clusterFeo = kk2.uglyCluster()

    knn2 = KNN(k=k, cast_float=True)
    knn2.saveData(train, label)


    #####
    # Grafico

    #steps = 200
    steps = 100

    xmin, xmax = train[:,0].min() - 1, train[:,0].max() + 1
    ymin, ymax = train[:,1].min() - 1, train[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)


    labels = knn2.predict(np.c_[xx.ravel(), yy.ravel()])

    z = labels.reshape(xx.shape)

    plt.figure()

    plt.contourf(xx, yy, z, 2, alpha=.4, cmap='jet')
    plt.scatter(train[:,0], train[:,1], c=label, cmap='jet', alpha=0.7)

    plt.xticks(())
    plt.yticks(())

    #plt.savefig('Informe/2/4_Lindo_k={}.pdf'.format(k), format='pdf', bbox_inches='tight')

    #plt.close(5)
    plt.show()





#---------------------------------
#			Ejercicio 5
#---------------------------------


class LinearClassifier():
    def __init__(self, n):
        self.n = n  #Numero de clases
        pass
    def fit(self, x_train, y_train, x_test, y_test, size_bacht=50, lr=1e-4, landa=1e-3, epochs=200, normalize=False):
        self.lr = lr
        self.l = landa
        self.epochs = epochs
        self.sbacht = size_bacht
        self.loss = np.array([])
        self.acc = np.array([])
        self.test_acc = np.array([])
        # Tuve algunos problemas con el pasaje de argumentos por referencia, por el momento esto lo arregla
        self.X = np.copy(x_train)       
        self.X_t = np.copy(x_test)

        # Acomodo las entradas del entrenamiento
        self.X = self.X.reshape(len(self.X), self.X[0].size).astype(np.float)
        self.X = np.hstack((np.ones((len(self.X),1)), self.X))              # Pongo 1 para el bias
        self.Y = y_train.reshape(y_train.size)

        # Acomodo las entradas del test
        self.X_t = x_test.reshape(len(x_test), x_test[0].size).astype(np.float)
        self.X_t = np.hstack((np.ones((len(self.X_t),1)), self.X_t))        # Pongo 1 para el bias
        self.Y_t = y_test.reshape(y_test.size)

        if(normalize):
            self.X   /= 255
            self.X_t /= 255

        # Inicializo W
        self.W = np.random.uniform(-10, 10, size=(self.n, self.X.shape[1]))

        n_bacht = int(len(x_train)/self.sbacht)  # Cuantos bacht tengo, pueden que me queden imagenes afuera. Arreglar

        # Recorro las epocas        
        for e in range(self.epochs):
            c_loss = 0
            c_acc = 0
            for i in range(n_bacht):

                x_bacht = self.X[self.sbacht*i: self.sbacht*(i+1)]      # Me quedo con un pedazo de los datos
                y_bacht = self.Y[self.sbacht*i: self.sbacht*(i+1)]

                loss, dw = self.loss_gradient(x_bacht, y_bacht)     # Me molesta que el VSCode me tire que esta linea esta mal

                c_loss += loss

                predict_y = self.predict(x_bacht)
                c_acc += self.accuracy(y_bacht, predict_y)
                
                self.W -= self.lr * dw
            
            if (e % 50 == 0):
                print(e,"/",self.epochs," ",c_acc/n_bacht)
            
            self.loss = np.append(self.loss, c_loss/n_bacht)        
            self.acc  = np.append(self.acc,   c_acc/n_bacht)       
            predict_test = self.predict(self.X_t)
            self.test_acc = np.append(self.test_acc, self.accuracy(self.Y_t, predict_test))
        

        print("Precision final con los datos de entrenamiento: ", self.acc[-1])
        print("Precision final con los datos de test: ", self.test_acc[-1])
        
        self.e = np.arange(self.epochs)

        #self.plotLoss()
        #self.plotAcurracy()

    def plotLoss(self):
        plt.plot(self.e, self.loss)
        plt.show()

    def plotAcurracy(self):
        plt.plot(self.e, self.loss)
        plt.show()

    def predict(self, x_testt):
        if(self.X.shape[1] != x_testt.shape[1]):
            x_testt = x_testt.reshape(len(x_testt), x_testt[0].size).astype(np.float)
            x_testt = np.hstack((np.ones((len(x_testt),1)), x_testt))
        scores = np.dot(self.W, x_testt.T)
        return np.argmax(scores, axis=0)

    def loss_gradient(self,X,Y):
        pass

    def accuracy(self, y_true, y_pred):
        accuracy = (np.sum(y_true == y_pred) / len(y_true))
        return accuracy*100
    
    def getLoss(self):      # Por si quiero graficar afuera
        return self.loss
    
    def getAccuracy(self):  # Por si quiero graficar afuera
        return self.acc
    
    def getAccTest(self):
        return self.test_acc[-1]
    
    def getAccuracyTest(self):
        return self.test_acc


class SVM(LinearClassifier):
    def __init__(self, n,delta=1):
        super(SVM, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        # Las dimensiones del batch ya deberian estar acomodadas

        scores = np.dot(self.W, X.T)         #Calculo los Scores

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,self.sbacht)

        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        # Intento restar cada score a su columna (ie imagen) correspondiente
        resta = scores - y_win[np.newaxis,:] + self.delta
        resta[Y, idx] = 0                   # y acomodo a 0 los que deberian ganar

        resta *= np.heaviside(resta,0)      # Mato todo lo que es negativo

        #resta[resta<0] = 0                  

        loss = (resta.sum() / self.sbacht) + 0.5 * self.l * np.sum(self.W * self.W)     # 

        #resta = np.heaviside(resta,0)

        resta[resta>0] = 1                  # Lo que queda es solo positivo y lo mando a 1

        resta[Y, idx] -= resta.sum(axis=0)[idx]     # El lugar debe ganar tiene la resta de todos los que le ganaron

        dw = (np.dot(resta, X) / self.sbacht ) + self.l * self.W

        return loss, dw
    

class Softmax(LinearClassifier):
    def __init__(self, n, delta=1):
        super(Softmax, self).__init__(n)
        self.delta = delta
    def loss_gradient(self, X, Y):
        super()

        scores = np.dot(self.W, X.T)                #Calculo los Scores

        scores = scores - scores.max(axis=0)        # Le resto el score maximo para que sea mas estable

        # Ok, tengo los scores. Quiero ver cual deberia ganar de cada columna (una columna es una imagen)
        # El primer indice es el Score que debe ganar y el segundo el numero de imagen
        idx = np.arange(0,Y.size)
        y_win = scores[Y, idx]          # Creo que con esto deberia poder quedarme con los scores que deben ganar

        exp = np.exp(scores)            # Tomo las exponenciales de cada Score 

        sumatoria = exp.sum(axis=0)         # Necesitamos la sumatoria de todas las exponenciales

        softmax_fun = exp * (1.0/sumatoria)     # Calculo la Softmax de cada imagen

        softmax_fun[Y, idx] -= 1               # Le resto la delta de Kronecker a las que tienen que ganar

        log_softmax = np.log(sumatoria) - y_win             # Calculamos el log Softmax de cada imagen
                                                            # Creo que no es estrictamente la softmax, pero se entiende

        loss = log_softmax.mean() + 0.5 * self.l * np.sum(self.W * self.W)      # Promedio todas las imagenes y sumo
                                                                                # regularizacion

        dw = (np.dot(softmax_fun, X) / self.sbacht ) + self.l * self.W      # Me hace ruido este producto, pero
                                                            # por las dimensiones es lo unico que tiene sentido

        return loss, dw




def ejercicio_5_Softmax(c='mnist', epocas=100):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    soft = Softmax(10)

    print("Sofmax - {}".format(c))

    soft.fit(x_train, y_train, x_test, y_test, epochs=epocas, normalize=False)

    e = np.arange(epocas)
    coste = soft.getLoss()

    plt.plot(e, coste)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/Loss_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    plt.close()
    plt.show()

    acc = soft.getAccuracy()

    plt.plot(e, acc)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/Accuracy_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

    acc_t = soft.getAccuracyTest()

    plt.plot(e, acc_t)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/AccuracyTest_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

def ejercicio_5_SVM(c='mnist', epocas=100):
    np.random.seed(10)


    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    svm = SVM(10)

    print("SVM - {}".format(c))

    svm.fit(x_train, y_train, x_test, y_test, epochs=epocas, normalize=False)

    e = np.arange(epocas)
    coste = svm.getLoss()

    plt.plot(e, coste)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/SVM_Loss_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

    acc = svm.getAccuracy()

    plt.plot(e, acc)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/SVM_Accuracy_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

    acc_t = svm.getAccuracyTest()

    plt.plot(e, acc_t)
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    #plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/SVM_AccuracyTest_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

def ejercicio_5_Comparar(c='mnist', epocas=100):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    ###### Softmax
    soft = Softmax(10)

    print("Sofmax - {}".format(c))

    soft.fit(x_train, y_train, x_test, y_test, epochs=epocas)

    softmax_coste = soft.getLoss()
    softmax_acc   = soft.getAccuracy()
    softmax_acc_test   = soft.getAccuracyTest()

    ###### SVM
    svm = SVM(10)

    print("SVM - {}".format(c))

    svm.fit(x_train, y_train, x_test, y_test, epochs=epocas)

    svm_coste = svm.getLoss()
    svm_acc   = svm.getAccuracy()
    svm_acc_test   = svm.getAccuracyTest()
    
    #### GRAFICO

    e = np.arange(epocas)

    plt.figure()

    plt.plot(e, softmax_coste, label=r"Softmax")
    plt.plot(e, svm_coste, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/Comparacion_Coste_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    #plt.close()
    plt.show()

    plt.figure()

    plt.plot(e, softmax_acc, label=r"Softmax")
    plt.plot(e, svm_acc, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Training", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/Comparacion_Precision_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    #plt.close()
    plt.show()

    plt.figure()

    plt.plot(e, softmax_acc_test, label=r"Softmax")
    plt.plot(e, svm_acc_test, label=r"SVM")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    plt.tight_layout()
    #plt.savefig('Informe/5/Comparacion_Precision_TEST_{}_e{}.pdf'.format(c,epocas), format='pdf', bbox_inches='tight')
    #plt.pause(1)
    #plt.close()
    plt.show()


def barridoParametros(method='SVM',c='mnist', epocas=200):
    np.random.seed(10)

    if(c == 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif(c == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    lear_rate = [1e-2, 1e-3, 1e-4, 1e-5]
    loss_g = []
    acc_g  = []
    acc_t  = []

    for i in lear_rate:
        if(method == 'SVM'):
            svm = SVM(10)
        elif(method == 'Softmax'):
            svm = Softmax(10)
        svm.fit(x_train, y_train, x_test, y_test, lr=i, epochs=epocas)

        loss_g += [svm.getLoss()]
        acc_g += [svm.getAccuracy()]
        acc_t += [svm.getAccuracyTest()]

    loss_g = np.array(loss_g)
    acc_g = np.array(acc_g)
    acc_t = np.array(acc_t)

    e = np.arange(epocas)

    plt.figure()
    plt.plot(e, loss_g[0], label=r"lr=1e-2")
    plt.plot(e, loss_g[1], label=r"lr=1e-3")
    plt.plot(e, loss_g[2], label=r"lr=1e-4")
    plt.plot(e, loss_g[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lr_loss_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()

    plt.figure()
    plt.plot(e, acc_g[0], label=r"lr=1e-2")
    plt.plot(e, acc_g[1], label=r"lr=1e-3")
    plt.plot(e, acc_g[2], label=r"lr=1e-4")
    plt.plot(e, acc_g[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lr_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()

    plt.figure()
    plt.plot(e, acc_t[0], label=r"lr=1e-2")
    plt.plot(e, acc_t[1], label=r"lr=1e-3")
    plt.plot(e, acc_t[2], label=r"lr=1e-4")
    plt.plot(e, acc_t[3], label=r"lr=1e-5")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lr_accTest_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()

    

    # Ahora pruebo variando lambda
    lambda_v = [1e-2, 1e-3, 1e-4]
    loss_g_2 = []
    acc_g_2  = []
    acc_t_2  = []

    for i in lambda_v:
        if(method == 'SVM'):
            svm = SVM(10)
        elif(method == 'Softmax'):
            svm = Softmax(10)
        svm.fit(x_train, y_train, x_test, y_test, landa=i, epochs=epocas)

        loss_g_2 += [svm.getLoss()]
        acc_g_2 += [svm.getAccuracy()]
        acc_t_2 += [svm.getAccuracyTest()]

    loss_g = np.array(loss_g)
    acc_g = np.array(acc_g)
    acc_t = np.array(acc_t)

    e = np.arange(epocas)

    plt.figure()
    plt.plot(e, loss_g_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, loss_g_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, loss_g_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Funcion de coste", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()

    plt.figure()
    plt.plot(e, acc_g_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, acc_g_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, acc_g_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()

    plt.figure()
    plt.plot(e, acc_t_2[0], label=r"$\lambda=1e-2$")
    plt.plot(e, acc_t_2[1], label=r"$\lambda=1e-3$")
    plt.plot(e, acc_t_2[2], label=r"$\lambda=1e-4$")
    plt.xlabel("Epoca", fontsize=15)
    plt.ylabel("Accuracy Test", fontsize=15)
    plt.legend(loc="best")
    #plt.savefig('Informe/5/{}_Barrido_lambda_acc_{}_e{}.pdf'.format(method,c,epocas), format='pdf', bbox_inches='tight')
    #plt.close()
    #plt.pause(1)
    plt.show()







if __name__ == "__main__":
    
    print("\nEjercicio 1")
    
    ejercicio_1()

    print("\nEjercicio 2 - el grafico se actualiza solo")
    
    ejercicio_2()

    print("\nEjercicio 3 - solo pruebo con los primeros 20 test.")
    print("Se puede descomentar en el main para probar con todo el set de datos")
   
    ejercicio_3_20test(c='mnist')

    ejercicio_3_20test(c='cifar10',norma=2,k=3)

    # Estos dos tardan bastante en correr
    #ejercicio_3_Alltest(c='mnist')
    #ejercicio_3_Alltest(c='cifar10')

    print("\nEjercicio 4")
 
    ejercicio4(k=1)

    ejercicio4(k=3)

    ejercicio4(k=7)

    ejercicio4_extra()

    print("\nEjercicio 5 - puede tardar un rato")
    #ejercicio_5_Softmax(c='mnist',epocas=200)
    #ejercicio_5_SVM(c='mnist',epocas=200)

    ejercicio_5_Comparar(c='mnist',epocas=250)
    ejercicio_5_Comparar(c='cifar10', epocas=300)

    # Estos 4 tardan bastante en correr
    #barridoParametros(method='SVM'    ,c='mnist'  ,epocas=250)
    #barridoParametros(method='Softmax',c='mnist'  ,epocas=250)
    #barridoParametros(method='SVM'    ,c='cifar10',epocas=250)
    #barridoParametros(method='Softmax',c='cifar10',epocas=250)
