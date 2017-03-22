# this code is designed for plotting the bungee jumper's velocity
# this code evaluates whether the velocity is above 36m/s
# or not when time is 4 seconds,

import numpy as np
import matplotlib.pyplot as plt

g=9.81
cd=0.25
t=4
v=36
m=np.linspace(40, 200, 100)
fm=np.sqrt(m*g/cd)*np.tanh(np.sqrt(g*cd/m)*t)-v
plt.figure(22)
plt.plot(m, fm)
plt.grid(True)
y1=np.linspace(0, 0, 100)
plt.plot(m, y1)


