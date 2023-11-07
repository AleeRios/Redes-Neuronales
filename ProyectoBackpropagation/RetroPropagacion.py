# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: REDES NEURONALES
Tema: Proyecto
Alumno: Rios Campusano Beckham Alejandro y Garduño Sanchez Bladimir
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementar el algoritmo de backpropagation en Python

Created on Tue Nov 23 19:12:47 2021

@author: alebe
"""

import numpy as np
    
class RetroPropagacion():
    
    def __init__(self, Entrada, Oculta, Salida):
        self.Entrada = Entrada + 1
        self.Oculta = Oculta
        self.Salida = Salida
        
      # Activa todos los nodos (vectores) de la red neuronal
        self.activacionEntrada = [1.]*self.Entrada
        self.activacionOculta = [1.]*self.Oculta
        self.activacionSalida = [1.]*self.Salida
        
        # Matriz Pesos sinapticos
        self.W1 = np.random.rand(self.Entrada, self.Oculta)
        self.W2 = np.random.rand(self.Oculta, self.Salida)
        
        
        # Finalmente construir factor de impulso (matriz)
        self.ci = np.random.rand(self.Entrada, self.Oculta)
        self.co = np.random.rand(self.Oculta, self.Salida)
        
    
    def sigmoide(self, z):
        return 1. / (1 + np.exp(-z))
    
    def dsigmoide(self, y):
        return 1.0 - y**2
            
    def Nodos(self):
        print('Todas las capas')
        print(self.activacionEntrada)
        print(self.activacionOculta)
        print(self.activacionSalida)
    
    def ActualizarPesos(self, entradas):
        if (len(entradas) != self.Entrada - 1):
            print("No es el mismo el numero de nodos que la entrada")
    
        # Activar la capa de entrada
        for i in range(self.Entrada-1):
            self.activacionEntrada[i] = entradas[i]
    
        # Activar la capa oculta
        for j in range(self.Oculta):
            sum = 0.0
            for i in range(self.Entrada):
                sum = sum + self.activacionEntrada[i] * self.W1[i][j]
            self.activacionOculta[j] = self.sigmoide(sum)
    
        # Activar la capa de salida
        for k in range(self.Salida):
            sum = 0.0
            for j in range(self.Oculta):
                sum = sum + self.activacionOculta[j] * self.W2[j][k]
            self.activacionSalida[k] = self.sigmoide(sum)
    
        return self.activacionSalida[:]
    
    def propagarAtras(self, N, M, tar):
        if len(tar) != self.Salida:
           raise ValueError("No coincide con el numero de nodos")
           
        delta1 = [0.] * self.Salida
        for i in range(self.Salida):
            err = tar[i] - self.activacionSalida[i]
            delta1[i] = self.dsigmoide(self.activacionSalida[i]) * err
        
        delta2 = [0.] * self.Oculta
        for i in range(self.Oculta):
            err = 0.
            for j in range(self.Salida):
                err = err + delta1[j] * self.W2[i][j]
            delta2[i] = self.dsigmoide(self.activacionOculta[i]) * err
            
        for i in range(self.Oculta):
            for j in range(self.Salida):
                c = delta1[j] * self.activacionOculta[i]
                self.W2[i][j] = self.W2[i][j] + N * c + M * self.co[i][j]
                self.co[i][j] = c
        
        for i in range(self.Entrada):
            for j in range(self.Oculta):
                c = delta2[j] * self.activacionEntrada[i]
                self.W1[i][j] = self.W1[i][j] + N * c + M * self.ci[i][j]
                self.ci[i][j] = c
        
        err = 0.
        for i in range(len(tar)):
            err = err + 0.5 * (tar[i] - self.activacionSalida[i])**2
        return err
    
    def entrenar(self, p, epoch = 300, N = .5, M = .1):
        for i in range(epoch):
            err = 0.
            for j in p:
                entradas = j[0]
                tar = j[1]
                self.ActualizarPesos(entradas)
                err = err + self.propagarAtras(N, M, tar)
            if i % 100 == 0:
                print('Error: ', err)
                