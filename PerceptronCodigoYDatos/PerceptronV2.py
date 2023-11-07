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
class PerceptronV2:
    '''
    Versión 2 del algoritmo del perceptrón, adaptada para 
    etiquetas 0 y 1
    '''
    
    def fit(self, X, Y, epochs=100):
        '''Entrena o ajusta los pesos sinápticos del perceptrón. X son los
        atributos (DataFrame), Y: Son las etiquetas 
        '''
        numElementos = X.shape[0]
        numAtributos = X.shape[1]
        # Agregar unos en la primera columna
        X = pd.concat([pd.Series(np.ones(numElementos)), X], axis=1)
        self.w = np.random.randn(numAtributos + 1) # Vector de pesos sinápticos aleatorio
        for k in range(epochs):
            for i in range(numElementos):
                xi = X.iloc[i, :]   # Atributo
                yi = Y[i]           # Etiqueta conocida
                y = self.predict(xi) # Calcula predicción
                self.w = self.w + (yi - y) * xi

    
    def predict(self, x):
        "Predice la etiqueta (0 o 1) para la muestra x (vector o matrix numpy)"
        if np.dot(self.w, x) >= 0:
            y = 1
        else:
            y = 0
        return y

