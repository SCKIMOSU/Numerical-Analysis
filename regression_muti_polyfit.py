import numpy as np
import matplotlib.pyplot as plt

# 구간 [0,4*pi]에 사인 곡선을 따라 균일한 간격의 점 100개를 생성합니다.
x = np.linspace(0,4*np.pi,100)
y=np.sin(x)

# 구간 [0,4*pi]에 따라 사인 곡선을 시각화한다
plt.figure(1)
plt.plot(x, y)

# np.polyfit을 사용하여 이들 점에 7차 다항식을 피팅합니다.

p = np.polyfit(x,y,7)

# 좀 더 촘촘한 그리드에서 다항식을 계산하고 결과를 플로팅합니다.
x1 = np.linspace(0, 4*np.pi)
y1 = np.polyval(p,x1)s
plt.figure(2)
plt.plot(x, y, x1, y1, 'r*')
plt.grid() 
