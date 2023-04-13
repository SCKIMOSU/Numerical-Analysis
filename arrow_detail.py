plt.figure(1)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
xx = np.linspace(-4, 4, 800)
yy = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(xx, yy)
Z = f2(X, Y)
levels=np.logspace(-1, 3, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="gray", levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000]
plt.plot(1, 1, 'ro', markersize=10)
x, y = 1.5, 1.5
g = f2g(x, y)
plt.arrow(x, y, -s * mu * g[0], -s * mu * g[1],
           head_width=0.04, head_length=0.04, fc='k', ec='k', lw=2)
