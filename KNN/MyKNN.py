#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
REDES NEURONALES
Tema: Aprendizaje Supervisado
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementación de algoritmo KNN

Created on Wed Oct  6 11:47:32 2021

@author: asdruballopezchau
"""
import numpy as np
class MyKNN:
    
    def fit(self, X, Y, K=5):
        '''
        Genera o calibra el modelo con los datos

        Parameters
        ----------
        X : DataFrame
            DESCRIPTION. Atributos del conjunto de datos
                        de entrenamiento
        Y : Series
            DESCRIPTION. Etiquetas del conjunto de datos
                        de entrenamiento

        Returns
        -------
        None.

        '''
        self.X = X
        self.Y = Y
        self.K = K

    
    def calculaDistancias(self, xi):
        '''
        Calcula la distancia entre xi y todas las instancias en self.X

        Parameters
        ----------
        xi : array numpy
            DESCRIPTION.

        Returns
        -------
        List
            DESCRIPTION. Todas las distancias en una lista

        '''
        X = self.X.values # Como arreglo de numpy
        #print(X[0]);
        distancias = []
        for xj in X:
            d = self.distanciaEuclideana(xi, xj)
            distancias.append(d)
        #print(distancias[0])
        return distancias
        
    
    def distanciaEuclideana(self, xi, xj):
        '''
        Calcula distancia Euclideana entre
        dos instancias

        Parameters
        ----------
        xi : array numpy
            DESCRIPTION. Instancia i
        xj : array numpy
            DESCRIPTION. Instancia j

        Returns
        -------
        float
            DESCRIPTION. Distancia Euclideana
                    entre xi y xj

        '''
        return np.sqrt(np.sum(np.power(xi - xj, 2)))

    def getIndicesKDistanciasMenores(self, distancias):
        indices = []
        for k in range(self.K):
            idx = distancias.index( np.min(distancias) )
            distancias[idx] = np.max(distancias)
            indices.append(idx)
        return indices
    
    def calcularClaseMasFrecuente(self, indices):
        for i in indices:
            print(self.Y[i])

    def predict(self, Xtest):
        '''
        Realiza la prediccion con K-NN

        Parameters
        ----------
        Xtest : DataFrame o array numpy
            DESCRIPTION. Atributos de instancias 

        Returns
        -------
        ypred : List
            DESCRIPTION. Las predicciones 

        '''
        ypred = []
        Xtest = Xtest.values

        for xi in Xtest:
            print(xi)
            distancias = self.calculaDistancias(xi)
            indices = self.getIndicesKDistanciasMenores(distancias)
            self.calcularClaseMasFrecuente(indices)
        
        return ypred

