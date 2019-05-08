import numpy as np
import matplotlib.pyplot as plt

x=np.array([0, 1, 2, 3, 4, 5.0])
y=np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
xp=np.linspace(0, 5, 100)

p2=np.polyfit(x, y, 2)
plt.figure(11)
plt.plot(x, y, 'ro-', x, np.polyval(p2, x), 'b*-')
plt.grid()

a=np.array([[6, 15, 55], [15, 55, 225], [55, 225, 979]])
b=np.array([152.6, 585.6, 2488.8])

x=np.linalg.solve(a, b)

# x= array([ 2.47857143,  2.35928571,  1.86071429])
# p2 = array([ 1.86071429,  2.35928571,  2.47857143])

y1=p2[0]*xp**2+p2[1]*xp+xp

plt.plot(xp, y1, 'xr-')
