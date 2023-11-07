#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: Redes Neuronales
Tema: Perceptron con salidas 0 y 1
Alumno: Tu Nombre
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementa el algoritmo del perceptron para conjuntos
 de datos con etiqueta 0 y 1

Created on Wed Sep  1 12:12:50 2021

@author: asdruballopezchau
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class PerceptronV2:
    '''
    Versión 2 del algoritmo del perceptrón, adaptada para 
    etiquetas 0 y 1
    '''
    def identifyClasses(self, Y):
        '''
        Identifica las clases existentes en la columna de etiquetas
        
        Parameters
        ----------
        Y: DataFrame de Pandas
           Columna de etiquetas.
        
        Returns
        -------
        Clases: list
           Regresa una lista de las etiquetas existentes.
        '''
        
        clases = list(set(Y))
        return clases
        
    def fit(self, X, Y, epochs=100):
        '''Entrena o ajusta los pesos sinápticos del perceptrón. X son los
        atributos (DataFrame), Y: Son las etiquetas 
        '''
        numElementos = X.shape[0]
        numAtributos = X.shape[1]
        # Agregar unos en la primera columna
        X = pd.concat([pd.Series(np.ones(numElementos)), X], axis=1)
        ajuste = self.adjust(self.identifyClasses(Y))
        self.ajuste=ajuste
        self.w = np.random.randn(numAtributos + 1) # Vector de pesos sinápticos aleatorio
        for k in range(epochs):
            for i in range(numElementos):
                xi = X.iloc[i, :]   # Atributo
                yi = Y[i]           # Etiqueta conocida
                y = self.__predict__(xi) # Calcula predicción
                self.w = self.w + (yi - y) * xi * ajuste
    
    def adjust(self, classes):
        '''
        Calculas el valor de la variable de ajuste en el algoritmo de entrenamiento
        w = w + (yi-y) * xi * ajuste
        ajuste = 1.0 si las etiquetas son 0 y 1
        ajuste = 0.5 si las etiquetas son -1 y 1

        Parameters
        ----------
        classes : list
            Las clases existentes en los datos.

        Returns
        -------
        ajuste : double
            Valor de ajuste para entrenamiento.

        '''
        suma = np.sum(classes)
        if suma == 0:
            ajuste = 0.5
        elif suma == 1:
            ajuste = 1
        return ajuste
    
    def __predict__(self, x):
        '''
        x ya tiene el 1 integrado

        Parameters
        ----------
        x : array de numpy
            Una muestra del conjunto de datos con el 1 ya agregado.

        Returns
        -------
        y: float
           Prediccion de la clase x.

        '''
        prod = np.dot(self.w, x)
        if self.ajuste == 1:
            if prod >= 0:
                y = 1
            else:
                y = 0
        elif self.ajuste == 0.5:
            if prod >= 0:
                y = 1
            else:
                y = -1
        return y
    
    def predict(self, X):
        '''
        Predice las etiquetas las muestras en X

        Parameters
        ----------
        X : DataFrame
            Contiene las muestras.

        Returns
        -------
        y : list
            Las predicciones para cada muestra.

        '''
        "Predice la etiqueta (0 o 1) para la muestra x (vector o matrix numpy)"
        X2 = X.values # Conversio a arreglo numpy
        unos = np.ones((len(X2), 1)) # Agrega los
        X2=np.append(unos, X2, axis=1)# 1´s necesarios al inicio
        predicciones = []
        for i in range(len(X2)):
            x = X2[i, :]
            y = self.__predict__(x)
            predicciones.append(y)
        return predicciones                  
    
    def plot(self, X, Y):
        '''
        Grafica el conjunto de datos si tiene dos varibles

        Parameters
        ----------
        X : DataFrame
            Atributos.
        Y : DataFrame / Serie pandas
            Clases.

        Returns
        -------
        None.

        '''
        clases = self.identifyClasses(Y)
        clase1 = clases[0]
        clase2 = clases[1]
        idx = Y == clase1
        xclase1 = X[idx]
        plt.plot(xclase1.iloc[:, 0],xclase1.iloc[:, 1], "ro")
        idx = Y == clase2
        xclase2 = X[idx]
        plt.plot(xclase2.iloc[:, 0],xclase2.iloc[:, 1], "sb")
        self.plotDecisionBoundary()
    
    def plotDecisionBoundary(self):
        w = list(self.w)
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        m = -w1/w2
        b = -w0/w2
        x = np.linspace(-1, 1, 100)
        y = (m*x) + b
        plt.plot(x, y, "-k")