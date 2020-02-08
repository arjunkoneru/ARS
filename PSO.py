# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:55:15 2020
"""
#### Imports Libraries ####
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
import seaborn as sns
import random
import decimal
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from functions import Rosenbrock, Rastrigin, Paraboloid, Easom, Eggholder
import time

#### Define functions from functions.py #####
functs = {
   # 'Gradient' : Gradient_Descent,
    '0' : Paraboloid,
    '1' : Rastrigin,
    '2' : Rosenbrock,
    '3' : Easom,
    '4' : Eggholder
}

######### Settings params
#### Ask User for input ####
no_particles = int(input("Number of particles - (default 20): \n >>") or 20) #20
max_iters = int(input("Number of iterations - (default 100): \n >>") or 100) #1000
F = functs[(input('Chose Paraboloid (0), Rastrigin (1), Rosenbrock (2), Easom (3) or Eggholder (4) - (default 0): \n >>') or '0')]
trueMin = [0,0]
if F.__name__ == 'Eggholder':
    interval = 600
    trueMin = [512, 404.2319]
elif F.__name__ == 'Easom':
    interval = int(input("Size of interval - (default 5): \n >>") or 5) #1000
    trueMin = [np.pi, np.pi]
else:
    interval = int(input("Size of interval - (default 5): \n >>") or 5) #1000
xdim = bool(input("3D plots? (0 / 1) (default 0): \n>>") or 0)

random.seed(0)
no_dimensions = 2
a = 2
b = 2
w1 = 0.9
w2 = 0.4


#### Initialize states, velocities, pbest, gbest, plot functionspace
state = np.zeros((no_particles,no_dimensions))
velocity = np.zeros((no_particles,no_dimensions))
particle_best_score = np.ones(no_particles)
particle_best_location = np.zeros((no_particles,no_dimensions))
x = np.linspace(-interval,interval,50)
y = np.linspace(-interval,interval,50)
x, y = np.meshgrid(x, y)
z = np.array(F([x, y]))
for i in range (no_particles):
    for d in range (no_dimensions):
        state[i][d] = round(random.uniform(-interval,interval), 2)
        velocity[i][d] = round(random.uniform(-20,20), 2)
    particle_best_score[i] = F(state[i])
    particle_best_location[i] = state[i]
        
global_best = np.min(F(state[:].T))
global_best_location = state[np.argmin(F(state[:].T))]

fig = plt.figure()  
if xdim:
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.plasma, alpha = 0.3)
else:
    ax = fig.subplots()
    surf = ax.contourf(x, y, z, rstride=1, cstride=1, cmap=cm.plasma, alpha = 0.3)
plt.ion()
plt.show()  


#### PSO ####
##### 3-D visualization ######
for k in range(max_iters):
    for i in range (no_particles):
        fitness_value = F(state[i])

        if fitness_value < particle_best_score[i]:
            particle_best_score[i] = fitness_value
            particle_best_location[i] = state[i]
        
        if fitness_value < global_best:
            global_best = fitness_value
            global_best_location = state[i]
    w = ((w1 - w2)*(max_iters -k-1)/max_iters) + w2
    velocity = np.clip((w * velocity) + (a * random.uniform(0,1) * (particle_best_location - state)) + (b * random.uniform(0,1) * (global_best_location - state)),-2,2)
    state = np.clip((state + velocity),-interval,interval)
    if ((k==0) or (k+1==max_iters) or (k==int((max_iters-1)/2))):
        if xdim:
            ax.scatter((state[:,0]), (state[:,1]), zs=F(state[:].T), zdir='z', s=30)
        else:
            ax.scatter((state[:,0]), (state[:,1]), s=5)
        plt.pause(2)
if xdim:    #3d
    ax.scatter(trueMin[0], trueMin[1], zs=F(trueMin), c='b')
else:       #2d
    ax.scatter(0,0, facecolors='none', edgecolors='r', marker='D')
#### End
print("Fit val:{}".format(fitness_value))
print(particle_best_location[0])
print(np.round(particle_best_location[0]))
plt.show()

