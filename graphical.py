import numpy as np
import matplotlib.pyplot as plt
g=9.8; cd=0.25; t=4

m=np.linspace(40, 200, 100)

v=np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)
v1=np.sqrt(g*m/cd)*np.tanh(np.sqrt(g*cd/m)*t)-36

k=np.linspace(0, 0, 100)

plt.figure(1)
plt.plot(m, v, 'r.')
plt.grid()
plt.show()

plt.figure(2)
#plt.plot(m, v1, 'b.')
plt.plot(m, v1, m, k)
plt.grid()
plt.show()
