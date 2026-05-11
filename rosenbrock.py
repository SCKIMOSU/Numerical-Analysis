import numpy as np


import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 부호 깨짐 방지
plt.rcParams['mathtext.fontset'] = 'dejavusans'   # ← 이 한 줄 추가

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def f2(x, y):
    return (1 - x)**2 + 100.0 * (y - x**2)**2

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f2(X, Y)

fig = plt.figure(figsize=(16, 5))

# === (1) Log-scale contourf — 가장 직관적 ===
ax1 = fig.add_subplot(1, 3, 1)
levels = np.logspace(-1, 3.5, 30)
cf = ax1.contourf(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
ax1.contour(X, Y, Z, levels=[0.5, 2, 10, 50, 200, 1000],
            colors='white', linewidths=0.6, alpha=0.6)
# 골짜기 곡선
xs = np.linspace(-2, 2, 200)
ax1.plot(xs, xs**2, 'w--', lw=1.5, label='valley $y=x^2$')
ax1.plot(1, 1, 'r*', markersize=18, label='min (1,1)')
ax1.plot(-1, -1, 'bo', markersize=10, label='start (-1,-1)')
ax1.set_title('Log-scale contour (banana valley)')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.legend(loc='upper left', fontsize=9)
plt.colorbar(cf, ax=ax1, label='f(x,y)')

# === (2) 3D 표면 (작은 영역) ===
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
xs = np.linspace(-2, 2, 80)
ys = np.linspace(-1, 3, 80)
Xs, Ys = np.meshgrid(xs, ys)
Zs = f2(Xs, Ys)
ax2.plot_surface(Xs, Ys, Zs, cmap='viridis', edgecolor='none', alpha=0.9)
ax2.set_title('3D surface (가파른 절벽)')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('f')
ax2.view_init(elev=35, azim=-60)

# === (3) 골짜기 단면 (y = x²을 따라) ===
ax3 = fig.add_subplot(1, 3, 3)
xs = np.linspace(-2, 2, 400)
ax3.plot(xs, f2(xs, xs**2),  'b-', lw=2, label='골짜기 바닥 (y=x²)')
ax3.plot(xs, f2(xs, xs**2 + 0.3), 'r--', lw=1.5, label='골짜기 위 (y=x²+0.3)')
ax3.plot(1, 0, 'r*', markersize=15)
ax3.set_yscale('log')
ax3.set_xlabel('x'); ax3.set_ylabel('f (log)')
#ax3.set_title('단면: 골짜기 바닥은 4차함수 (1−x)⁴')
ax3.set_title(r'단면: 골짜기 바닥은 4차함수 $(1-x)^{4}$')
ax3.legend(); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()
