# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:55:15 2020

@author: ammar
"""

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
import seaborn as sns
import random
import decimal
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from functions import Rosenbrock, Rosenbrock_plt, Rastrigin , Rastrigin_plt
import time


functs = {
   # 'Gradient' : Gradient_Descent,
    'Rastrigin'  : Rastrigin,
    'Rosenbrock' : Rosenbrock
}


no_particles = int(input("Number of particles (default 20): \n >>") or 20) #20
max_iters = int(input("Number of iterations (default 100): \n >>") or 100) #1000
interval = int(input("Size of interval (default 5): \n >>") or 5) #1000
F = functs[(input('Chose Rastrigin or Rosenbrock (default): \n >>') or 'Rosenbrock')]
#F_plt = 

np.random.seed = 0
no_dimensions = 2

#print(F)

random.seed(0)
#no_particles = 20
no_dimensions = 2


a = 2
b = 2
w1 = 0.9
w2 = 0.4
state = np.zeros((no_particles,no_dimensions))
velocity = np.zeros((no_particles,no_dimensions))
particle_best_score = np.ones(no_particles)# * 1000
particle_best_location = np.zeros((no_particles,no_dimensions))




x = np.linspace(-interval,interval,200)#, num = no_particles)
y = np.linspace(-interval,interval,200)#, num = no_particles)
x, y = np.meshgrid(x, y)
#print("x: {} \ny: {}".format(x,y))

z = np.array(F([x, y]))
#print("Z: {}".format(z))

for i in range (no_particles):
    for d in range (no_dimensions):
        #state[i][d] = float(decimal.Decimal(random.randrange(-500, 500))/100)
        #velocity[i][d] = float(decimal.Decimal(random.randrange(-200, 200))/100)
        state[i][d] = round(random.uniform(-interval,interval), 2)
        velocity[i][d] = round(random.uniform(-2,2), 2)
        

global_best = np.min(F(state))#[0],state[i][1])
global_best_location = state[np.argmin(F(state))]
#print("glob: {}".format(global_best_location))

#print("Velocities:\n{}".format(velocity))


fig = plt.figure()  
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.plasma, alpha = 0.3)
#surf = ax.contourf(x, y, z, cmap=cm.plasma)
plt.ion()
plt.show()  
for k in range(max_iters):
    for i in range (no_particles):
        fitness_value = F(state[i])

        if fitness_value < particle_best_score[i]:
            particle_best_score[i] = fitness_value
            particle_best_location[i] = state[i]
#            print("New Particle best: {}\nNew Particle Best Location: {}".format(particle_best_score, particle_best_location))

        if fitness_value < global_best:
            global_best = fitness_value
            global_best_location = state[i]
#            print("New global best: {}\nNew global Best Location: {}".format(global_best,global_best_location))
        #if k % 20 == 0:
            #ax.scatter(state[i][0], state[i][1], zs=F(state[i]), zdir='z', c='green', s=100)
#        print(i)

            #print(state[i])
        w = ((w1 - w2)*(max_iters -k)/max_iters) + w2
        #velocity[i] = (w * velocity[i]) + (a * random.uniform(0,1) * (particle_best_location[i] - state[i])) + (b * random.uniform(0,1) * (global_best_location - state[i]))
        velocity[i] = np.clip((w * velocity[i]) + (a * random.uniform(0,1) * (particle_best_location[i] - state[i])) + (b * random.uniform(0,1) * (global_best_location - state[i])), -2,2)
        #state[i] = state[i] + velocity[i]
        state[i] = np.clip((state[i] + velocity[i]),-interval,interval)
#print("state[0]: {}\nstate[0][:]: {}\nstate[1]: {}\nstate[0][0]: {}\nstate[1][0]: {}".format(state[:],state[:][0], state[1], state[0][0], state[1][0]))
    ax.scatter((state[:,0]), (state[:,1]), zs=F(state[:].T), zdir='z', s=100)
    plt.pause(1)
print("Fit val:{}".format(fitness_value))
print(particle_best_location[0])
print(np.round(particle_best_location[0]))

#ax.view_init(elev=30., azim=35)
plt.show()

