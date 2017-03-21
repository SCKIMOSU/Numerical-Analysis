import numpy as np
import matplotlib.pyplot as plt

g=9.8
cd=0.25
m=68

v0=0
v1=(1-0)*(g-cd/m*v0**2)+v0
v2=(2-1)*(g-cd/m*v1**2)+v1
v3=(3-2)*(g-cd/m*v2**2)+v2
v4=(4-3)*(g-cd/m*v3**2)+v3

time=np.arange(0, 5)
vel=np.array([v0, v1, v2, v3, v4])


#  original method

vel_o=np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*time)


plt.figure(1)
plt.plot(time, vel, '-b1', label='euler')
plt.plot(time,  vel_o, '-ro', label='differential')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('velocity by euler and differential')
plt.show()

