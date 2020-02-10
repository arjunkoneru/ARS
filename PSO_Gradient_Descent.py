# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:43:21 2020
"""

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
import seaborn as sns
import random
import decimal
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from functions import Rosenbrock, dRosenbrock, Rastrigin, dRastrigin, Paraboloid, dParaboloid
import time

class Gradient_Descent:
    
    def __init__ (self, F, dF, interval, nr_iteration, init_state, trueMin):
        self.interval = interval
        self.F = F
        self.dF = dF
        self.nr_iteration = nr_iteration
        trueMin = trueMin
        state_gd = init_state
        
    
        learning_rate = 0.001 # After experiment, 0.0002 works good.
#        state_gd = np.array([round(random.uniform(-interval,interval), 2),round(random.uniform(-interval,interval), 2)])
        #state_ga = [ 4.02656064 -2.254784  ]
       # nr_iteration=100
        gd_history=[]
        
        x = np.linspace(-interval,interval,500)
        y = np.linspace(-interval,interval,500)
        x, y = np.meshgrid(x, y)
        try:
            z = np.array(F([x, y]))
            fig = plt.figure()
            ax = fig.subplots()
            surf = ax.contourf(x, y, z, cmap=cm.plasma, alpha = 0.3)
#            
            ax.scatter((state_gd[:,0]), (state_gd[:,1]), s=50)
            for i in range(nr_iteration):
                for j in range(len(state_gd)):
                #    state_gd = np.clip(state_gd ,-interval,interval)
                    f = (state_gd[j][0]**2 + 100 * (state_gd[j][1] - state_gd[j][0]**2)**2)
                    gd_history.append([state_gd[j],f])
                    fi = np.array(dRastrigin(state_gd[j]))
                    state_gd[j] = state_gd[j] - np.dot(learning_rate,fi)
                    if i%10==0:
                        print(state_gd[j])
                        ax.scatter((state_gd[j][0]), (state_gd[j][1]), s=50)
                plt.pause(0.1)
                    
            ax.scatter(trueMin[0], trueMin[1], facecolors='none', edgecolors='r', marker='D', s=50 )
            plt.show()
            
            print("After {} tierations, the minimus is {} at {}".format(nr_iteration, gd_history[-1][1], gd_history[-1][0]))
        except:
            print("Derivative of {} not yet implemented!".format(F.__name__))
    