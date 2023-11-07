#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
REDES NEURONALES
Tema: Aprendizaje Supervisado
Profesor: Dr. Asdrúbal López Chau
Descripción: Pruebas al algoritmo K-NN

Created on Wed Oct  6 13:08:26 2021

@author: asdruballopezchau
"""

import pandas as pd
import random
from MyKNN import MyKNN

datos = pd.read_csv("iris.csv")
# Separar en conujunto de datos de prueba y de entrenamiento
# 30% pruebas, 70% entrenamiento

indices = list(range(len(datos)))
random.shuffle(indices) # Revuelve los indices
# Separa en conjunto de datos de entrenamiento
j = int(0.7*len(datos))
idx = indices[ 0: j]
train = datos.iloc[idx, :]

# Separa en conjunto de datos de prueba
idx = indices[j:]
test = datos.iloc[idx, :]

train.reset_index(inplace=True)
test.reset_index(inplace=True)

clf = MyKNN()
# Separa en atributos y clases
Xtrain  = train.iloc[:, 0:-1]
Ytrain = train.iloc[:, -1]
clf.fit(Xtrain, Ytrain, K=1)
# Separa en atributos y clases
Xtest = test.iloc[:, 0:-1]
Ytest = test.iloc[:, -1]
clf.predict(Xtest.iloc[0,:])
