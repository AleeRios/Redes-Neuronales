#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
REDES NEURONALES

Tema: BACKPROPAGATION
Profesor: Dr. Asdrúbal López Chau
Descripción: Cálculo del error 

Created on Wed Nov 10 11:52:27 2021

@author: asdruballopezchau
"""

import numpy as np

def lineal(x):
    return x

# Rectifier Linear Unit
def ReLu(x):
    '''xn = []
    for item in x:
        xn.append(np.max(0., item))
    xn = np.array(xn)
    return xn
    '''
    return [np.max([0., item]) for item in x]

def sigmoid(x):
    return 1./(1 + np.exp(-x))

# matrices de pesos sinápticos
W1 = np.random.randn(3, 4)
W2 = np.random.randn(2, 4)
W3 = np.random.randn(2, 3)
# Entrada xn = [8, 7, 6]
xn = np.reshape(np.array([8., 6.0, 7.0]), (3, 1))
# Salida real yn = [1, 0]
yn = np.reshape([1., 0.], (2, 1)) 
# Agregar un uno a xn
equis = [1.]
equis.extend(np.reshape(xn, (-1)))
equis = np.reshape(np.array(equis), (4, 1))
a2 = np.matmul(W1, equis)
############
a2conUno = [1.]
a2conUno.extend(np.reshape(a2, (-1)))
a2conUno = np.reshape(a2conUno , (4, 1))
#############
a3 = np.reshape(np.matmul(W2, a2conUno), (-1))
a3 = ReLu(a3)
a3 = np.reshape(a3, (2, 1))
#############
a3conUno = [1.]
a3conUno.extend(np.reshape(a3, (-1)))
a3conUno = np.reshape(a3conUno, (3, 1))
y_est = np.reshape(np.matmul(W3, a3conUno), (-1))
y_est = sigmoid(y_est)
y_est = np.reshape(y_est, (2, 1))
#print(y_est)
en = 0.5 * np.sum(np.power(yn - y_est, 2.0))
print(en)


