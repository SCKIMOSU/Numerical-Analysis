import matplotlib.pyplot as plt
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
