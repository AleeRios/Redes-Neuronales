# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: REDES NEURONALES
Tema: EXAMEN 1ER PARCIAL
Alumno: Rios Campusano Beckham Alejandro
Profesor: Dr. Asdrúbal López Chau
Descripción: Modifica el código fuente del perceptrón para que sea capaz de 
utilizar la función deactivación logística

Created on Wed Sep 22 13:40:17 2021

@author: alebe
"""

import pandas as pd
import numpy as np
from PerceptronLogistic import PerceptronLogistic

def pruebaA():
    train = pd.read_csv("trainA.csv") # Lectura de los datos
    
    X = train.iloc[:, 0:2]
    Y = train.iloc[:, -1]
    clf = PerceptronLogistic()
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
    
pruebaA()