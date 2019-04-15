import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

x1=np.linspace(0,10,50)
x2=np.linspace(-10,10,50)
y1=x1**3-9*x1**2+24*x1-7
y2=x2**3-9*x2**2+24*x2-7
plt.figure(1)
plt.plot(x1, y1, 'ro-')
plt.grid()
plt.show()
plt.figure(2)
plt.plot(x2, y2, 'b*-')
plt.grid()
plt.show()

fx=lambda x: x**3-9*x**2+24*x-7
x=fsolve(fx, 1)
print("Real Root= ", x)
