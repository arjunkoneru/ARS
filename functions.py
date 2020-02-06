# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:51:19 2020

@author: ammar
"""
import numpy as np


def Rosenbrock(x):
    return (x[0]**2 + 100 * (x[1] - x[0]**2)**2)

def Rosenbrock_plt(x, y):
    return (x**2 + 100 * (y - x**2)**2)

def Rastrigin(x):
    return ((10 * 2) + sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(2)))

def Rastrigin_plt(x, y):
    return ((10 * len(x)) + sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))))