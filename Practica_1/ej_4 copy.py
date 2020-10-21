#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 01-09-2020
File: ej_4.py
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
#from keras.datasets import cifar10
#from keras.datasets import mnist

import seaborn as snn

snn.set(font_scale = 1)

class KNN():

    def __init__(self, k=1, orden=2):
        self.k = k
        self.orden = orden
    
    def norma(self,x1, x2, o):
        return np.linalg.norm(x1 - x2, ord=o)
    
    def saveData(self, dataX, dataY):
        self.X = dataX.reshape(len(dataX), dataX[0].shape).astype(np.float)
        #self.X = dataX.reshape(len(dataX), dataX[0].size).astype(np.float)
        self.Y = dataY.reshape(dataY.size)
    
    def predict(self,X):
        #X = X.reshape(len(X), X[0].size).astype(np.float)
        X = X.reshape(len(X), X[0].shape).astype(np.float)
        y_pred = np.array([])
        # y_pred = [self._predict(x) for x in X]        # Hago el for en vez de esto xq quiero ver cuanto avanza
        for i in range(len(X)):
            if (((i % 100) == 0) and len(X) >= 500):
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


class createData():
    def __init__(self, mean_, std_, label):
        self.mean = mean_
        self.std = std_
        self.label = label
    
    def getPoints(self, n):
        data = np.random.normal(self.mean, self.std, size=(n,2))
        label = np.full(n, self.label)
        return data, label

class clusterEj2():
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




np.random.seed(19)

n = 7
    
ej4 = clusterEj2()
x_train, y_train, x_test, y_test = ej4.nClusters(n)

for i in range(n):
    plt.scatter(x_train[i][:,0], x_train[i][:,1,], marker='.')

#plt.pause(3)
#plt.close()
plt.show()






"""                 TENGO QUE VER ESTO
clstr1 = createData([0,0], 1, 0)
clstr2 = createData([3,3], 1, 1)
clstr3 = createData([3,-3], 1, 2)
clstr4 = createData([-3,-3], 1, 3)
clstr5 = createData([-3,3], 1, 4)

train_1, predict_1 = clstr1.getPoints(100)
train_2, predict_2 = clstr2.getPoints(100)
train_3, predict_3 = clstr3.getPoints(100)
train_4, predict_4 = clstr4.getPoints(100)
train_5, predict_5 = clstr5.getPoints(100)

plt.scatter(train_1[:,0], train_1[:,1])
plt.scatter(train_2[:,0], train_2[:,1])
plt.scatter(train_3[:,0], train_3[:,1])
plt.scatter(train_4[:,0], train_4[:,1])
plt.scatter(train_5[:,0], train_5[:,1])
plt.show()

train = np.concatenate([train_1, train_2, train_3, train_4, train_5])
predict = np.concatenate([predict_1, predict_2, predict_3, predict_4, predict_5])


knn = KNN(7,2)

knn.saveData(train, predict)


# Ahora genero datos de test
test_1, check_1 = clstr1.getPoints(50)
test_2, check_2 = clstr2.getPoints(50)
test_3, check_3 = clstr3.getPoints(50)
test_4, check_4 = clstr4.getPoints(50)
test_5, check_5 = clstr5.getPoints(50)

test = np.concatenate([test_1, test_2, test_3, test_4, test_5])
check = np.concatenate([check_1, check_2, check_3, check_4, check_5])

result = knn.predict(test)

print("Precision para norma ", 2, "y k = ", 1, ": ", accuracy(result, check) )



#####
# Grafico

steps = 300

xmin, xmax = train[:,0].min() - 1, train[:,0].max() + 1
ymin, ymax = train[:,1].min() - 1, train[:,1].max() + 1
x_span = np.linspace(xmin, xmax, steps)
y_span = np.linspace(ymin, ymax, steps)
xx, yy = np.meshgrid(x_span, y_span)


labels = knn.predict(np.c_[xx.ravel(), yy.ravel()])

z = labels.reshape(xx.shape)



#colores = ['g', 'b', 'r', 'm', 'y', 'b']

colores = np.linspace(0,1,5)

cols = [colores[i] for i in predict]
cols2 = [colores[i] for i in predict]

plt.figure()
plt.axes([0.04, 0.04, 0.99, 0.99])

plt.contourf(xx, yy, z, 5, alpha=.3, cmap='jet')

#plt.scatter(train[:,0], train[:,1], c=predict, cmap='jet', alpha=0.6)
plt.scatter(test[:,0], test[:,1], c=check, marker='x', cmap='jet', alpha=0.6)

plt.show()
"""


"""

TENGO QUE ACOMODAR EL CREATECLUSTER DEL 2

TIPO DE DATO FLOAT
"""


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





kk2 = createCluster(2)
train, label = kk2.uglyData()
clusterFeo = kk2.uglyCluster()


knn2 = KNN(k=3)
knn2.saveData(train, label)


#####
# Grafico

steps = 300

xmin, xmax = train[:,0].min() - 1, train[:,0].max() + 1
ymin, ymax = train[:,1].min() - 1, train[:,1].max() + 1
x_span = np.linspace(xmin, xmax, steps)
y_span = np.linspace(ymin, ymax, steps)
xx, yy = np.meshgrid(x_span, y_span)


labels = knn2.predict(np.c_[xx.ravel(), yy.ravel()])

z = labels.reshape(xx.shape)



plt.figure()
#plt.axes([0.04, 0.04, 0.99, 0.99])

plt.contourf(xx, yy, z, 3, alpha=.3, cmap='jet')

plt.scatter(train[:,0], train[:,1], c=label, cmap='jet', alpha=0.6)
plt.savefig('Informe/2/4_Lindo_k=3.pdf', format='pdf', bbox_inches='tight')
#plt.scatter(test[:,0], test[:,1], c=check, marker='x', cmap='jet', alpha=0.6)

plt.close(5)







kk2 = createCluster(2)
train, label = kk2.uglyData()
clusterFeo = kk2.uglyCluster()


knn2 = KNN(k=7)
knn2.saveData(train, label)


#####
# Grafico

steps = 300

xmin, xmax = train[:,0].min() - 1, train[:,0].max() + 1
ymin, ymax = train[:,1].min() - 1, train[:,1].max() + 1
x_span = np.linspace(xmin, xmax, steps)
y_span = np.linspace(ymin, ymax, steps)
xx, yy = np.meshgrid(x_span, y_span)


labels = knn2.predict(np.c_[xx.ravel(), yy.ravel()])

z = labels.reshape(xx.shape)



plt.figure()
#plt.axes([0.04, 0.04, 0.99, 0.99])

plt.contourf(xx, yy, z, 3, alpha=.3, cmap='jet')

plt.scatter(train[:,0], train[:,1], c=label, cmap='jet', alpha=0.6)
plt.savefig('Informe/2/4_Lindo_k=7.pdf', format='pdf', bbox_inches='tight')
#plt.scatter(test[:,0], test[:,1], c=check, marker='x', cmap='jet', alpha=0.6)

plt.show()





"""