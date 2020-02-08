# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:51:19 2020
"""
import numpy as np


def Rosenbrock(x):
    return (x[0]**2 + 100 * (x[1] - x[0]**2)**2)

def Rastrigin(x):
#    return ((10 * 2) + (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))
    return ((10 * 2) + sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) for i in range(len(x))))

def Paraboloid(x):
    return (x[0]**2/0.001) +(x[1]**2/0.001)

def Easom(x):
    return (-np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2)))

def Eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(np.abs((x[0]/2) + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47)))))