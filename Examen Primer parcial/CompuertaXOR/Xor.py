# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: REDES NEURONALES
Tema:EXAMEN 1ER PARCIAL
Alumno: Rios Campusano Beckham Alejandro
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementa la compuerta XOR con neuronas tipo Mc-Culloch Pitts

Created on Wed Sep 22 12:30:34 2021

@author: alebe
"""

from Compuertas import compuertaAND,compuertaOR,compuertaNOT,compuertaXOR
import pandas as pd

def pruebaXor():
    not1=compuertaNOT()
    not2=compuertaNOT()
    and1=compuertaAND()
    and2=compuertaAND()
    or1=compuertaOR()
    xor1=compuertaXOR()
    X=pd.read_csv("pruebaXOR.csv")
    print(X,end="\n\n")
    
    for i in range(len(X)):
        x1=X.iloc[i,0:1]
        x2=X.iloc[i,1:2]
        wx1=int(x1)
        wx2=int(x2)
        enot1=not1.predict(wx1*-3.7)
        enot2=not2.predict(wx2*-3.7)
        eand1=and1.predict((enot1*.7)+(wx2*.5))
        eand2=and2.predict((wx1*.7)+(enot2*.5))
        sOR=or1.predict((eand1*.2)+(eand2*.1))
        print("Prediccion: ",xor1.predict(sOR*1),end="\n\n")

pruebaXor()