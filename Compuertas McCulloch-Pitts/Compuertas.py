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

Created on Tue Aug 31 15:14:54 2021

@author: alebe
"""

class compuertaAND:
    
    def __init__(self):
        self.w=None
        
    def heaviside(self,y):
        if y>1:
            return 1
        else:
            return 0
        
    def predict(self,y):
        return self.heaviside(y)

class compuertaOR:
    
    def __init__(self):
        self.w=None
        
    def heaviside(self,y):
        if y>0:
            return 1
        else:
            return 0
    
    def predict(self,y):
        return self.heaviside(y)

class compuertaNOT:
    
    def __init__(self):
        self.w=None
        
    def heaviside(self,y):
        if y>-1:
            return 1
        else:
            return 0
    
    def predict(self,y):
        return self.heaviside(y)