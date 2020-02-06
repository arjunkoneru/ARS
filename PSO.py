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
from functions import Rosenbrock, Rosenbrock_plt, Rastrigin, Rastrigin_plt


functs = {
   # 'Gradient' : Gradient_Descent,
    'Rastrigin'  : Rastrigin,
    'Rosenbrock' : Rosenbrock
}


no_particles = int(input("Number of particles (default 20): \n >>") or 20) #20
max_iters = int(input("Number of iterations (default 1000): \n >>") or 1000) #1000
F = functs[(input('Chose Rastrigin or Rosenbrock (default): \n >>') or 'Rosenbrock')]

np.random.seed = 0
no_dimensions = 2

print(F)

random.seed(0)
#no_particles = 20
no_dimensions = 2

interval=5

a = 2
b = 2
w1 = 0.9
w2 = 0.4
state = np.zeros((no_particles,no_dimensions))
velocity = np.zeros((no_particles,no_dimensions))
particle_best_score = np.ones(no_particles)# * 1000
particle_best_location = np.zeros((no_particles,no_dimensions))




x = np.linspace(-interval,interval, num = no_particles)
y = np.linspace(-interval,interval, num = no_particles)
x, y = np.meshgrid(x, y)
z = F([x, y])

for i in range (no_particles):
    for d in range (no_dimensions):
        #state[i][d] = float(decimal.Decimal(random.randrange(-500, 500))/100)
        #velocity[i][d] = float(decimal.Decimal(random.randrange(-200, 200))/100)
        state[i][d] = round(random.uniform(-5,5), 2)
        velocity[i][d] = round(random.uniform(-2,2), 2)
        

global_best = F(state[i])
global_best_location = state[i]

print("Velocities:\n{}".format(velocity))


fig = plt.figure()  
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.plasma, alpha=0.3)
plt.ion()
plt.show()  
for k in range(max_iters):
    for i in range (no_particles):
        fitness_value = F(state[i])

        if fitness_value < particle_best_score[i]:
            particle_best_score[i] = fitness_value
            particle_best_location[i] = state[i]
        if fitness_value < global_best:
            global_best = fitness_value
            global_best_location = state[i]
        if k % 25 == 0:
            if state[i][0] <= 5 and state[i][1] <= 5:
                ax.scatter(state[i][0], state[i][1], zs=Rosenbrock(state[i])+2, zdir='z', c='green')
                plt.pause(1)
            #print(state[i])
        w = ((w1 - w2)*(max_iters -k)/max_iters) + w2
        #velocity[i] = (w * velocity[i]) + (a * random.uniform(0,1) * (particle_best_location[i] - state[i])) + (b * random.uniform(0,1) * (global_best_location - state[i]))
        velocity[i] = np.clip((w * velocity[i]) + (a * random.uniform(0,1) * (particle_best_location[i] - state[i])) + (b * random.uniform(0,1) * (global_best_location - state[i])), -2,2)
        #state[i] = state[i] + velocity[i]
        state[i] = np.clip((state[i] + velocity[i]),-5,5)

print("Fit val:{}".format(fitness_value))
print(particle_best_location[0])
print(np.round(particle_best_location[0]))

#ax.view_init(elev=30., azim=35)
plt.show()

