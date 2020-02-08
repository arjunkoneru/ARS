import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
import seaborn as sns
import random
import decimal
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
#from functions import Rosenbrock, Rastrigin, Paraboloid, Easom, Eggholder
import time

#a=0, b=100
def Rosenbrock(x):
    return (x[0]**2 + 100 * (x[1] - x[0]**2)**2)

def DerrivRosenbrock (x):
    dx = 2*x[0] - 400*x[0]*(x[1] - (x[0]**2))
    dy = 200*(x[1] - (x[0]**2))
    return dx, dy

interval=5

learning_rate=0.0001
state_gd = np.array([round(random.uniform(-interval,interval), 2),round(random.uniform(-interval,interval), 2)])
nr_iteration=100
gd_history=[]


x = np.linspace(-interval,interval,50)
y = np.linspace(-interval,interval,50)
x, y = np.meshgrid(x, y)
z = np.array(Rosenbrock([x, y]))

fig = plt.figure()
ax = fig.subplots()
surf = ax.contourf(x, y, z, cmap=cm.plasma, alpha = 0.3)


for i in range(nr_iteration):
#    state_gd = np.clip(state_gd ,-interval,interval)
    f = (state_gd[0]**2 + 100 * (state_gd[1] - state_gd[0]**2)**2)
    gd_history.append([state_gd,f])
    fi = np.array(DerrivRosenbrock(state_gd))
    state_gd = state_gd - np.dot(learning_rate,fi)
    if i%10==0:
        print(state_gd)
        ax.scatter((state_gd[0]), (state_gd[1]), s=5)
        plt.pause(2)
        
    plt.show()    