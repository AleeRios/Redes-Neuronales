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

def pruebaA():
    train = pd.read_csv("trainA.csv") # Lectura de los datos
    
    X = train.iloc[:, 0:2]
    Y = train.iloc[:, -1]
    clf = PerceptronV2()
    clf.fit(X, Y, epochs=10)
    test = pd.read_csv("testA.csv")
    Xtest = test.iloc[:, 0:2]
    predict = clf.predict(Xtest)
    clf.plot(X,Y)
    yreal = test.iloc[:, -1].values
    ypred = np.array(predict)
    bien = np.sum(yreal==ypred)
    mal = np.sum((yreal==ypred)==False)
    print("Clases idenfificadas: ", clf.identifyClasses(Y))
    print("Predicciones correctas: ", bien)
    print("Predicciones incorrectas: ", mal)

def pruebaB():
    train = pd.read_csv("trainB.csv") # Lectura de los datos
    
    X = train.iloc[:, 0:2]
    Y = train.iloc[:, -1]
    clf = PerceptronV2()
    clf.fit(X, Y, epochs=10)
    test = pd.read_csv("testB.csv")
    Xtest = test.iloc[:, 0:2]
    predict = clf.predict(Xtest)
    clf.plot(X,Y)
    yreal = test.iloc[:, -1].values
    ypred = np.array(predict)
    bien = np.sum(yreal==ypred)
    mal = np.sum((yreal==ypred)==False)
    print("Clases idenfificadas: ", clf.identifyClasses(Y))
    print("Predicciones correctas: ", bien)
    print("Predicciones incorrectas: ", mal)

pruebaA()
print("\n\n")
pruebaB()