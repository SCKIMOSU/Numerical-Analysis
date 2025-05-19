import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('TkAgg')  # 또는 'QtAgg', 'Agg'도 가능


# 샘플 데이터 생성
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([2.3, 2.5, 3.7, 3.9, 4.2, 4.4, 3.8, 3.1, 2.7])

# 다항 회귀 모델 (2차)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# 예측 및 시각화
x_test = np.linspace(1, 9, 100).reshape(-1, 1)
x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)

plt.scatter(x, y, color='blue', label='Data')
plt.plot(x_test, y_pred, color='red', label='Polynomial Fit')
plt.title("Polynomial Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
