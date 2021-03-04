import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

z=np.arange(-5, 5, 100)

x = tf.constant([0.0, 1.0, 50.0, 100.0])



import sympy as sp
x=sp.Symbol('x')
fx=3*x**2
print("F(x)=", sp.integrate(fx, x))


def f(x, y):
    return 2 * x**2 + 6 * x * y + 7 * y**2 - 26 * x - 54 * y + 107

xx = np.linspace(1, 16, 100)
yy = np.linspace(-3, 6, 90)
X, Y = np.meshgrid(xx, yy)
Z = f(X, Y)


def gx(x, y):
    return 4 * x + 6 * y - 26

def gy(x, y):
    return 6 * x + 14 * y - 54

xx2 = np.linspace(1, 16, 15)
yy2 = np.linspace(-3, 6, 9)
X2, Y2 = np.meshgrid(xx2, yy2)
GX = gx(X2, Y2)
GY = gy(X2, Y2)
plt.figure(figsize=(10, 5))
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10))
plt.quiver(X2, Y2, GX, GY, color='blue', scale=400, minshaft=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title("퀴버 플롯(quiver plot)")
plt.show()
