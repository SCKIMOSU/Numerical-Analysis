import numpy as np
import matplotlib.pyplot as plt

# 2차원 로젠브록 함수
def f2(x, y):
    return (1 - x)**2 + 100.0 * (y - x**2)**2

xx = np.linspace(-4, 4, 800)
yy = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(xx, yy)
Z = f2(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) 히트맵 (가장 빠름)
im = axes[0].imshow(Z, extent=[-4, 4, -3, 3], origin='lower',
                    cmap='viridis', aspect='auto')
axes[0].set_title('imshow')
plt.colorbar(im, ax=axes[0])

# 2) 등고선 (filled)
cf = axes[1].contourf(X, Y, Z, levels=30, cmap='viridis')
axes[1].contour(X, Y, Z, levels=15, colors='k', linewidths=0.3)
axes[1].set_title('contourf + contour')
plt.colorbar(cf, ax=axes[1])

# 3) 등고선 (라인만)
cs = axes[2].contour(X, Y, Z, levels=20, cmap='viridis')
axes[2].clabel(cs, inline=True, fontsize=8)
axes[2].set_title('contour')
axes[2].set_aspect('equal')

for ax in axes:
    ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# 800x600은 무거우니 다운샘플
ax.plot_surface(X[::5, ::5], Y[::5, ::5], Z[::5, ::5],
                cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f2(x,y)')
plt.show()
