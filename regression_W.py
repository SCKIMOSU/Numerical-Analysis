import numpy as np
import matplotlib.pyplot as plt

X = np.array([1., 2., 3.])
Y = np.array([1., 2., 3.])
m = len(X)

# W를 -3.0에서 5.0까지 0.1 간격으로 (총 81개 점)
W_val = np.linspace(-3.0, 5.0, 81)

# 브로드캐스팅으로 한 번에 cost 계산
#   hypothesis: shape (81, 3) — 각 W에 대해 W*X
#   cost: shape (81,)         — 각 W에 대한 MSE
hypothesis = W_val[:, None] * X[None, :]
cost_val   = np.sum((hypothesis - Y)**2, axis=1) / m

# 출력
for w, c in zip(W_val, cost_val):
    print(f'{w:5.1f}, {c:6.2f}')

# 시각화
plt.plot(W_val, cost_val)
plt.xlabel('W')
plt.ylabel('cost')
plt.grid(True)
plt.show()
