import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# line plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# ax = fig.gca(projection='3d')
theta = np.linspace(-4*np.pi,4*np.pi,100)
z = np.linspace(-2,2,100)
r = z**2+1
x = r*np.sin(theta)
y = r*np.cos(theta)

ax.plot(x,y,z,label='parametric curve')
ax.scatter(x,y,z)
ax.legend()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.tight_layout()
plt.show()
