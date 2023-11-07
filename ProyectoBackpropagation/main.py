# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTÓNOMA DEL ESTADO DE MÉXICO
CU UAEM ZUMPANGO
UA: REDES NEURONALES
Tema: Proyecto
Alumno: Rios Campusano Beckham Alejandro y Garduño Sanchez Bladimir
Profesor: Dr. Asdrúbal López Chau
Descripción: Implementar el algoritmo de backpropagation en Python

Created on Fri Nov 26 19:45:23 2021

@author: alebe
"""

import RetroPropagacion as re

ret = re.RetroPropagacion(2, 2, 1)

a = [[[0, 0], [0]],
       [[0, 1], [1]],
       [[1, 0], [1]],
       [[1, 1], [0]]]

ret.entrenar(a)