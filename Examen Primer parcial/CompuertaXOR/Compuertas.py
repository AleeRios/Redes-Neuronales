# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: EXAMEN 1ER PARCIAL
Tema: LABORATORIO PERCEPTRÓN
Alumno: Rios Campusano Beckham Alejandro
Profesor: Dr. Asdrúbal López Chau
Descripción:  Implementa la compuerta XOR con neuronas tipo Mc-Culloch Pitts

Created on Wed Sep 22 12:10:45 2021

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
    
class compuertaXOR:
    
    def __init__(self):
        self.w=None
    
    def heaviside(self,y):
        if y==1:
            return 1
        else:
            return 0
    
    def predict(self,y):
        return self.heaviside(y)
        