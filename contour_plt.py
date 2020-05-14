import numpy as np
import matplotlib.pyplot as plt



def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(1)
plt.contour(X, Y, Z, colors='black')

plt.figure(2)
plt.contour(X, Y, Z, 20, cmap='RdGy')

plt.figure(3)
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()

plt.figure(4)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')

plt.figure(5)
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
plt.colorbar()
