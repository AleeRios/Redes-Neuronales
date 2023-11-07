# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: REDES NEURONALES
Tema: LABORATORIO PERCEPTRÓN
Alumno: Rios Campusano Beckham Alejandro
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementa la compuerta AND, OR, NOT usando el modelo de una 
neurona tipo McCulloch-Pitts.

Created on Tue Aug 31 16:59:18 2021

@author: alebe
"""

from Compuertas import compuertaAND,compuertaOR,compuertaNOT
import pandas as pd

def pruebaAnd():
    and1=compuertaAND()
    X=pd.read_csv("pruebaAND.csv")
    w1=.7
    w2=.6
    print(X,end="\n\n")
    
    for i in range(len(X)):
        x1=X.iloc[i,0:1]
        x2=X.iloc[i,1:2]
        wx1=int(x1)*w1
        wx2=int(x2)*w2
        print("Predicción: ",and1.predict(wx1+wx2),end="\n\n")       

def pruebaOR():
    or1=compuertaOR()
    X=pd.read_csv("pruebaOR.csv")
    w1=.2
    w2=.1
    print(X,end="\n\n")
    
    for i in range(len(X)):
        x1=X.iloc[i,0:1]
        x2=X.iloc[i,1:2]
        wx1=int(x1)*w1
        wx2=int(x2)*w2
        print("Predicción: ",or1.predict(wx1+wx2),end="\n\n")

def pruebaNOT():
    not1=compuertaNOT()
    X=pd.read_csv("pruebaNOT.csv")
    w1=-3.2
    print(X,end="\n\n")
    
    for i in range(len(X)):
        x1=X.iloc[i,0:1]
        wx1=int(x1)*w1
        print("Predicción: ",not1.predict(wx1),end="\n\n")

pruebaAnd()
pruebaOR()
pruebaNOT()