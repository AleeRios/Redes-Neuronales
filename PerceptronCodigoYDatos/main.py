#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: Redes Neuronales
Tema: Perceptron con salidas 0 y 1
Alumno: Tu Nombre
Profesor: Dr. Asdrúbal López Chau
Descripción: Pruebas a perceptrónV2


Created on Wed Sep  1 12:21:33 2021

@author: asdruballopezchau
"""
import pandas as pd
import numpy as np
from PerceptronV2 import PerceptronV2

datos = pd.read_csv("datos.csv") # Lectura de los datos

# Número de columnas en un dataFrame:
# len(list(datos.columns))
# datos.shape[1] 
# Separar los atributos de las etiquetas
X = datos.iloc[:, 0:2]
Y = datos.iloc[:, -1]
clf = PerceptronV2()
clf.fit(X, Y, epochs=10)
prueba = np.array([1, -0.23, 0.3])
prediccion = clf.predict(prueba)
print('prediction=', prediccion)

prueba = np.array([1, 0.39, -0.15])
prediccion = clf.predict(prueba)
print('prediction=', prediccion)

prueba = np.array([1, 0.19, -1.0])
prediccion = clf.predict(prueba)
print('prediction=', prediccion)